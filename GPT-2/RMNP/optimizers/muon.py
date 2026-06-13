"""
Adapted from KellerJordan/modded-nanogpt: https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt2.py
"""

import torch
import torch.distributed as dist
import os
import time

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

def compute_diagonal_dominance(g):
    """
    计算 g @ g.T 的对角占优指标
    g: tensor of shape (m, n)

    对于 Gram 矩阵 V_t V_t^T 的第 i 行，定义对角占优度为：
    u_i = (V_t V_t^T)_{ii} - sum_{j != i} |(V_t V_t^T)_{ij}|
        = ||V_{t,i:}||_2^2 - sum_{j != i} |V_{t,i:} (V_{t,j:})^T|
    """
    # 对 tall matrix 转置，与 NS 一致，在 min(m,n) 维度计算
    if g.size(0) > g.size(1):
        g = g.T
    g = g.float()
    gram = g @ g.T  # shape: (min(m,n), min(m,n))
    m = gram.shape[0]
    if m < 2:
        keys = ["u_avg","u_min","u_max","ratio_to_max_avg","ratio_to_max_min",
                "ratio_to_max_max","ratio_to_avg_avg","ratio_to_avg_min","ratio_to_avg_max"]
        return {k: 0.0 for k in keys}

    # 对角线元素: (V_t V_t^T)_{ii} = ||V_{t,i:}||_2^2
    diag = torch.diag(gram)  # shape: (m,)

    # 非对角线元素绝对值之和: sum_{j != i} |(V_t V_t^T)_{ij}|
    off_diag_sum = torch.sum(torch.abs(gram), dim=1) - torch.abs(diag)  # shape: (m,)

    # u_i = diag_i - off_diag_sum_i
    u = diag - off_diag_sum  # shape: (m,)

    # 对每一行，计算非对角元素的 max 和 avg
    gram_abs = torch.abs(gram)
    mask = torch.eye(m, dtype=torch.bool, device=gram.device)

    # 每行非对角元素的最大值
    gram_for_max = gram_abs.clone()
    gram_for_max[mask] = float('-inf')
    off_diag_max_per_row = gram_for_max.max(dim=1).values  # shape: (m,)

    # 每行非对角元素的平均值: (行和 - 对角元素) / (m-1)
    off_diag_avg_per_row = (gram_abs.sum(dim=1) - diag.abs()) / (m - 1 + 1e-10)  # shape: (m,)

    # 每行的 ratio: diag_i / max(off_diag_i), diag_i / avg(off_diag_i)
    ratio_to_max = diag / (off_diag_max_per_row + 1e-10)  # shape: (m,)
    ratio_to_avg = diag / (off_diag_avg_per_row + 1e-10)  # shape: (m,)

    return {
        # 原有的 u 指标
        "u_avg": u.mean().item(),
        "u_min": u.min().item(),
        "u_max": u.max().item(),
        # 新增: diag / max(off_diag) 对 m 行取 avg/min/max
        "ratio_to_max_avg": ratio_to_max.mean().item(),
        "ratio_to_max_min": ratio_to_max.min().item(),
        "ratio_to_max_max": ratio_to_max.max().item(),
        # 新增: diag / avg(off_diag) 对 m 行取 avg/min/max
        "ratio_to_avg_avg": ratio_to_avg.mean().item(),
        "ratio_to_avg_min": ratio_to_avg.min().item(),
        "ratio_to_avg_max": ratio_to_avg.max().item(),
    }

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5, weight_decay=0.):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        # Timing for orthogonalization
        self.orth_time_step = 0.0
        self.orth_time_total = 0.0
        # Diagonal dominance metrics
        self.dd_u_bar_avg = 0.0
        self.dd_u_bar_min = 0.0
        self.dd_u_bar_max = 0.0
        self.dd_per_param = {}  # 每个参数的对角占优指标
        # 保存 g tensors 用于可视化
        self._save_g_this_step = False
        self._save_g_count = 0  # 每步保存的 g 数量上限 (0=全部)
        self._saved_g_tensors = {}  # param_idx -> g tensor (cpu)
        if 'WORLD_SIZE' in os.environ:
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.rank = int(os.environ['RANK'])
        else:
            self.world_size = 1
            self.rank = 0

    def step(self):
        self.orth_time_step = 0.0
        # 用于记录所有参数的对角占优指标
        all_u_avg, all_u_min, all_u_max = [], [], []
        self.dd_per_param = {}  # 清空每步的记录
        self._saved_g_tensors = {}  # 清空
        param_idx = 0  # 参数计数器

        for group in self.param_groups:

            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % int(self.world_size) == int(self.rank):
                    g = p.grad
                    assert g is not None
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    # 计算对角占优指标 (在 orthogonalization 之前)
                    dd_metrics = compute_diagonal_dominance(g)
                    all_u_avg.append(dd_metrics["u_avg"])
                    all_u_min.append(dd_metrics["u_min"])
                    all_u_max.append(dd_metrics["u_max"])
                    # 记录每个参数的指标
                    self.dd_per_param[f"param_{param_idx}"] = dd_metrics
                    # 保存 g tensor (正交化之前)
                    if self._save_g_this_step:
                        # 检查是否达到保存上限
                        if self._save_g_count == 0 or len(self._saved_g_tensors) < self._save_g_count:
                            self._saved_g_tensors[param_idx] = {
                                'g': g.detach().float().cpu(),
                                'param_index': i,  # 在 param_groups 中的全局 index
                                'shape': tuple(g.shape),
                                'dd': dd_metrics,
                            }
                    param_idx += 1

                    torch.cuda.synchronize()
                    _start = time.perf_counter()
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    torch.cuda.synchronize()
                    self.orth_time_step += time.perf_counter() - _start
                    g *= max(1, g.size(0)/g.size(1))**0.5
                    updates_flat[curr_idx:curr_idx+p.numel()] = g.flatten()
                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            if self.world_size > 1:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.mul_(1.-lr*weight_decay).add_(g, alpha=-lr)
                curr_idx += p.numel()
        self.orth_time_total += self.orth_time_step

        # 计算全局平均对角占优指标（跨 rank 同步）
        if len(all_u_avg) > 0:
            # 当前 rank 的局部统计
            local_sum_avg = sum(all_u_avg)
            local_sum_min = sum(all_u_min)
            local_sum_max = sum(all_u_max)
            local_count = len(all_u_avg)

            if self.world_size > 1:
                # 跨 rank 同步：收集所有 rank 的 sum 和 count
                local_tensor = torch.tensor([local_sum_avg, local_sum_min, local_sum_max, local_count],
                                           device='cuda', dtype=torch.float32)
                dist.all_reduce(local_tensor, op=dist.ReduceOp.SUM)
                global_sum_avg, global_sum_min, global_sum_max, global_count = local_tensor.tolist()
            else:
                global_sum_avg = local_sum_avg
                global_sum_min = local_sum_min
                global_sum_max = local_sum_max
                global_count = local_count

            # 计算全局平均
            self.dd_u_bar_avg = global_sum_avg / global_count
            self.dd_u_bar_min = global_sum_min / global_count
            self.dd_u_bar_max = global_sum_max / global_count

        # 重置保存标志
        self._save_g_this_step = False

    def enable_save_g(self, count=0):
        """标记下一次 step() 保存 g tensors

        Args:
            count: 保存的 g 数量上限，0 表示全部保存
        """
        self._save_g_this_step = True
        self._save_g_count = count

    def save_g_tensors(self, save_dir, iter_num):
        """
        将 rank 0 保存的 g tensors 写入磁盘。
        只保存当前 rank 处理的参数的 g (分布式中每个 rank 处理不同参数)。
        文件名: g_iter_{iter_num}_rank_{rank}.pt
        """
        if not self._saved_g_tensors:
            return
        g_dir = os.path.join(save_dir, 'g_tensors')
        os.makedirs(g_dir, exist_ok=True)
        save_path = os.path.join(g_dir, f'g_iter_{iter_num}_rank_{self.rank}.pt')
        torch.save({
            'iter_num': iter_num,
            'rank': self.rank,
            'world_size': self.world_size,
            'g_tensors': self._saved_g_tensors,
        }, save_path)
        print(f"[Rank {self.rank}] Saved {len(self._saved_g_tensors)} g tensors to {save_path}")

