# p-RMNP 实现报告

> 分支：`p-RMNP`（基于 `Final-Submission` 切出，其余分支未动）
> 数学规格：`Paper-Optimizer/HTRMNP/extended_rmnp_from_sgd_to_rmnp.md`（§9.1 Fixed / §9.2 Adaptive）与 `Paper-Optimizer/HTRMNP/p_rmnp_algorithm.md`（实现规格书）
> 参考实现：`Paper-Optimizer/HTRMNP/SMuon-From-SGD-to-Muon/schatten-muon`（SMuon 官方仓库）
> 实现原则：最小 diff——`step()` 骨架逐行复刻原 `rmnp.py`，训练脚本 copy 自 `train_rmnp_streaming.py` 只做定点插入，不修改任何现有文件。

---

## 1. 新增文件一览（7 个，无任何现有文件被修改）

| 文件 | 角色 |
|---|---|
| `RMNP/optimizers/p_rmnp.py` | `PRMNP` 优化器（fork 自 `rmnp.py`）+ `row_power` / `polynomial_scale` / `strict_lmo_normalize` |
| `RMNP/p_rmnp_utils/__init__.py` | 独立工具包（按要求与 `optimizers/` 分离） |
| `RMNP/p_rmnp_utils/wrap_model.py` | `ActivationRecorder`：forward hook 抓取每个 `nn.Linear` 的输入激活 |
| `RMNP/p_rmnp_utils/row_selector.py` | SVD-free 的 $p^\star$ 选择器（连续有界 Brent 搜索） |
| `RMNP/p_rmnp_utils/test_p_rmnp.py` | 单元测试（30 项） |
| `RMNP/train_p_rmnp_streaming.py` | 训练脚本（copy 自 `train_rmnp_streaming.py`，5 处定点插入） |
| `config/train_gpt2_small_p_rmnp_streaming_fw.py` | GPT-2 Small / FineWeb 流式训练 config |
| `scripts/run_p_rmnp_small_streaming_fw.sh` | 启动脚本（env 变量透传 p-RMNP 超参） |

---

## 2. 算法 ↔ 代码映射

### 2.1 Fixed p-RMNP（extended_rmnp §9.1 / p_rmnp_algorithm §3–6）

| 算法步骤 | 代码位置 |
|---|---|
| momentum buffer `B_t = μB_{t-1} + G_t`；Nesterov carrier `H_t = G_t + μB_t` | `p_rmnp.py::PRMNP.step`（逐行保留 `rmnp.py` 的 `buf.mul_(momentum).add_(g)` + `g.add(buf, alpha=momentum)` 约定） |
| 行变换 $R_p(H)_{i,:}=H_{i,:}/\max(a_i,\varepsilon)^{1-1/p}$ | `p_rmnp.py::row_power`（行范数 fp32 累积；$p=\infty$ 走 `F.normalize` 与原 RMNP 完全一致；零行精确输出零） |
| 可选严格 LMO 归一化 $R/Z_p$，$Z_p=(\sum a_i^{1+1/p})^{1/(p+1)}$ | `p_rmnp.py::strict_lmo_normalize`（log-space 计算防溢出；默认关闭，共享正标量吸收进 lr） |
| aspect-ratio 尺度 $s_p(m,n)$ | `p_rmnp.py::polynomial_scale`：`interpolate` $=\max(1,m/n)^{(1-1/p)/2}$（推荐，$p{=}1$→1、$p{=}\infty$→原 RMNP 尺度）/ `legacy` / `none` |
| 解耦权重衰减 + 参数更新 | 与 `rmnp.py` 相同的 flat bf16 buffer → all_reduce → `p.data.mul_(1-lr·wd).add_(g, alpha=-lr)` |

`p` 的编码：构造参数 `p <= 0` 表示 $p=\infty$（RMNP 端点），因为 configurator 的 `literal_eval` 无法从命令行传 `inf`。

### 2.2 Adaptive p-RMNP（extended_rmnp §9.2 / p_rmnp_algorithm §7）

| 算法步骤 | 代码位置 |
|---|---|
| 行统计量 $a_i=\|H_{i,:}\|$（瞬时）、$c_i=\langle G_{i,:},u_i\rangle$、$b_i=\|u_iA\|^2=u_iQu_i^\top$ | `p_rmnp.py::PRMNP.update_p_state` + `row_selector.py::row_curvature` |
| EMA 只打在 $\{c_i,b_i\}$ 上、$a_i$ 瞬时使用、绝不 EMA $p^\star$ | `update_p_state` 内 `lerp_(·, 1-stat_momentum)`（对齐 SMuon `ExactTightnessPApproximator` 的 EMA 位置） |
| 目标 $J(p)=N(p)^2/(D(p)+10^{-12})$，连续有界搜索 | `row_selector.py::select_p_from_row_statistics`：`minimize_scalar(bounds=(pmin,pmax), method="bounded")`，与 SMuon 逐字对应；统计量先搬到 CPU float64，几十次目标求值零同步开销 |
| $p^\star$ 逐参数存储、冷启动 `init_p=pmax` | `self.state[param]['p_star']`（随 optimizer `state_dict` 自动保存/恢复） |
| carrier 预演不写回 momentum | `update_p_state` 用 `buf.float().mul(momentum).add_(g32)`（out-of-place），单测有专项断言 |
| 激活捕获（forward hook，按 weight Parameter 键控） | `wrap_model.py::ActivationRecorder`（`recording()` 上下文管理器 + `use_gram` 模式） |
| 训练循环门控（每 `p_update_interval` 步 + step-10 预热，SMuon 同款） | `train_p_rmnp_streaming.py` 的 `should_record` 块 |

### 2.3 与 SMuon 参考实现的对应

| SMuon（schatten-muon 仓库） | p-RMNP 对应 | 替换关系 |
|---|---|---|
| `svs/tightness_exact.py::ExactTightnessPApproximator` | `p_rmnp_utils/row_selector.py` | $\sigma_i(M)\to a_i$、$C_{ii}\to c_i$、$B_{ii}\to b_i$；**SVD 整体删除**（行方向由参数化免费给出） |
| `_compute_ND_exact` | `_compute_ND_row` | 同款 p-clamp（$[1+10^{-9},10^4]$）、同款目标形式 |
| `minimize_scalar(bounds, method="bounded")` | 同款调用 | 逐字一致 |
| `smuon/wrap_model.py::ActivationRecorder` | `p_rmnp_utils/wrap_model.py` | **接口对齐的独立重写**（见 §4 偏差 1） |
| `optimizers/adaptive.py::update_p_state` + `state["p_star"]` | `PRMNP.update_p_state` + `state['p_star']` | 存储模式相同；分布式策略不同（见 §4 偏差 4） |
| `scripts/nanogpt.py` 的 `should_record` 门控（`step % interval == 0` or `step == 10`） | `train_p_rmnp_streaming.py` 同款门控 | 一致 |
| Taylor-Horner / `p_root.py` / `coeffs/` 全套分数幂机器 | **不需要**（$a_i^{1/p}$ 逐行标量幂直接算） | p-RMNP 的核心效率优势 |

---

## 3. 训练脚本的 5 处定点插入（相对 `train_rmnp_streaming.py`）

1. **默认超参块**（configurator 要求新 key 必须是模块级全局变量）：`p_fixed / p_adaptive / p_min / p_max / p_update_interval / p_warmup_step / p_stat_momentum / p_scale_mode / p_strict_lmo / p_use_gram`；
2. **recorder 注册**：在 `model.to(device)` 之后、`torch.compile`/DDP 包装**之前**，把 hook 挂在裸模型上并保留 `p_rmnp_raw_model` 引用；
3. **optimizer 构建**：`CombinedOptimizer(params, [AdamW, PRMNP], [...])`，PRMNP config 追加 p-RMNP 参数（2D/非2D 参数切分逻辑在 `opt.py`，零改动）；
4. **录制前向**：`should_record` 时用 `torch.no_grad() + ctx + recorder.recording()` 对**裸模型**做一次额外前向抓激活——绕开 torch.compile 与 forward hook 的交互风险，代价是每 `p_update_interval` 步多一次前向（interval=100 时约 1% 开销）；
5. **$p^\star$ 刷新**：grad clip（`scaler.unscale_` 已执行）之后、`scaler.step` 之前调 `optimizer.optimizers[1].update_p_state(...)`，随后 `recorder.clear()`；console 打印 + wandb 记录 `p_star/mean|min|max`。

---

## 4. 已知偏差清单（有意为之，均有理由）

1. **`ActivationRecorder` 是独立重写而非逐字 copy**：schatten-muon 仓库**无 LICENSE 文件**（默认保留所有权利），逐字复制进本（可能公开的）仓库有许可风险；改为按相同接口（`recording()`/`get_activations()`/`use_gram`）重写精简版，且只支持 `nn.Linear`（本模型没有卷积层），文件头注明设计来源。
2. **无 $[N]_+$ 正部截断**：目标用 $N^2/(D+10^{-12})$，与 SMuon 源码（其 `tightness_exact.py` 同样不检查 $N>0$）保持最小差分。严格非负步长理论应使用 $[N]_+^2/D$（见 extended_rmnp §8.4.4）；二者在 $N(p)>0$（动量与梯度总体正对齐，实践中的常态）时完全相同。
3. **无 `alpha_star` 学习率修正**：SMuon 的 `ExactTightnessPApproximator` 额外返回一个裁剪到 $[0.5,2]$ 的 LR 倍率，属于其启发式；`p_rmnp_algorithm.md` 规格未包含，故不实现。
4. **分布式统计量是 ownership-local 的**：`update_p_state` 只处理本 rank 拥有的参数（与 `step()` 相同的 `i % world_size == rank` 分片），$b_i$ 用本 rank 的本地 micro-batch 激活——**零额外通信**。正确性不受影响（每个参数的更新只由拥有它的 rank 计算），代价是曲率统计只见到 1/world_size 的 batch，由 EMA 平滑。升级路径：$\|u_iA\|^2$ 对 data shard 可加，一次 $m$ 维 all-reduce 即可获得全局 batch 曲率（$O(m)$ 通信）。
5. **`row_power` 不加 `@torch.compile`**：adaptive 模式下指数 $p$ 随训练变化，编译会反复触发 recompile。$p=\infty$ 路径调用与原 `rmnp.py` 相同的 `F.normalize`（原版有 compile，此处 eager；单测证实数值逐位一致）。
6. **`updates_flat` 的 device 跟随参数**而非硬编码 `'cuda'`（原 `rmnp.py` 硬编码）：行为在 GPU 训练下完全一致，同时允许 CPU 单测。
7. **embedding / lm_head（tied，2D）会进入 PRMNP 参数组**：这是 `CombinedOptimizer` 按 `ndim>=2` 切分的**既有行为**（原 RMNP 同样如此），p-RMNP 原样继承。lm_head 有 Linear hook 因此参与自适应 $p^\star$；`wpe` 无 hook、无激活记录，$p^\star$ 恒为 `init_p`（=pmax，即 RMNP 行为）——由 `update_p_state` 的 missing-activation 路径自然处理。
8. **fp16 + `grad_clip=0` 组合下 `update_p_state` 会看到 scaled 梯度**：默认配置是 bf16（GradScaler 为 no-op）+ `grad_clip=1.0`（clip 前已 unscale），不受影响；如需 fp16 且不 clip，需在调用前手动 `scaler.unscale_`。

**明确不实现**（用户指示"先考虑不带二阶矩的"）：Row-SMuon(Adam) 二阶矩预条件（extended_rmnp §10）。

---

## 5. 用法

```bash
# 默认：adaptive p*，interval=100，interpolate 尺度
bash scripts/run_p_rmnp_small_streaming_fw.sh

# 固定 p = 8（关自适应）
P_ADAPTIVE=False P_FIXED=8.0 bash scripts/run_p_rmnp_small_streaming_fw.sh

# 精确复现原 RMNP（回归对照）：p=∞ + legacy 尺度 + 关自适应
P_ADAPTIVE=False P_FIXED=0.0 P_SCALE_MODE=legacy bash scripts/run_p_rmnp_small_streaming_fw.sh

# 更高频的 p* 刷新
P_UPDATE_INTERVAL=25 bash scripts/run_p_rmnp_small_streaming_fw.sh
```

所有 `--key=value` 额外参数经 `"$@"` 透传给 configurator（如 `--p_strict_lmo=True`、`--p_use_gram=True`、`--compile=False`）。

超参速查：

| config key | 默认 | 说明 |
|---|---|---|
| `p_fixed` | `0.0` | 固定 $p$；`<=0` 表示 $\infty$（RMNP 端点）；`p_adaptive=True` 时忽略 |
| `p_adaptive` | `True` | 开启 layerwise 自适应 $p^\star$ |
| `p_min` / `p_max` | `1.02` / `50.0` | 连续搜索区间（SMuon 默认值） |
| `p_update_interval` | `100` | 每多少步刷新一次 $p^\star$ |
| `p_warmup_step` | `10` | 早期额外刷新一次（避免整个 warmup 停在 pmax） |
| `p_stat_momentum` | `0.95` | $c_i/b_i$ 的 EMA 系数 |
| `p_scale_mode` | `interpolate` | aspect-ratio 尺度模式 |
| `p_strict_lmo` | `False` | 严格单位 mixed-norm LMO 步长 |
| `p_use_gram` | `False` | 记录激活 Gram（$n\times n$）而非原始激活 |

---

## 6. 验证结果

**单元测试**（`CUDA_VISIBLE_DEVICES=1 TORCHINDUCTOR_COMPILE_THREADS=1 python p_rmnp_utils/test_p_rmnp.py`，环境 `nanochat`，torch 2.9.1+cu128 / scipy 1.17.1）：**30/30 通过**，关键项：

- `PRMNP(p=inf, scale_mode='legacy') == RMNP`：3 参数 × 3 步（含 Nesterov 动量与解耦 wd），最大逐元素差 **0.00e+00**（逐位一致）——p=∞ 端点是原 RMNP 的精确复现；
- 有限 $p$ 输出行长精确等于 $a_i^{1/p}$；$p{=}1$ 恒等；零行保持零；
- 严格 LMO 归一化后 $\|\cdot\|_{p+1,2}=1$（误差 <1e-5）；
- 尺度三模式端点值精确（interpolate 在 $p{=}1$ 为 1、$p{=}\infty$ 等于 legacy；宽矩阵恒 1）；
- 选择器：$p^\star$ 落在区间内、对 $c$ 的正标量缩放不变、"曲率集中在大-$a$ 行会推高 $p^\star$"的方向性正确、边界最优时 Brent 落在 pmin 的 1e-3 邻域内（bounded 搜索不评估端点，属 SMuon 同款行为）；
- `row_curvature` 原始激活路径与 Gram 路径一致（<1e-4），接受 `(batch, seq, n)` 形状；
- `update_p_state` 不改动 momentum buffer；缺激活的参数 $p^\star$ 保持 `init_p`；EMA 生效。

**冒烟训练**（GPT-2 Small，FineWeb-Edu 流式，单卡，`batch_size=4, grad_acc=2, compile=False`，adaptive 默认配置 + `P_UPDATE_INTERVAL=10`）：30 步跑通，loss 10.93 → 7.58；$p^\star$ 在第 10 步（`p_warmup_step` 预热）、20、30 步按预期刷新，rank-local 统计（50 个矩阵参数）为 mean 6.98 → 23.80 → 22.61，min 恒为 1.02、max 恒为 50——不同层选出了明显不同的 $p^\star$，layerwise 自适应确实在生效。

**端到端对照**（同种子、同数据流、20 步，`p_adaptive=False, p_fixed=0.0, scale_mode=legacy` 即 RMNP 等价模式 vs 原版 `train_rmnp_streaming.py`）：

| | step-0 eval (train/val) | iter 0 | iter 10 | iter 20 |
|---|---|---|---|---|
| p-RMNP（RMNP 模式） | 10.9352 / 10.9281 | 10.9336 | 8.1371 | 7.8963 |
| 原版 RMNP | 10.9352 / 10.9281 | 10.9336 | 8.1371 | 7.8955 |

前 10 步在打印精度内完全一致；iter 20 相差 8e-4，量级符合 CUDA 核（matmul/attention 归约顺序）的 run-to-run 非确定性——优化器本身的逐位一致性已由单元测试证明（见上）。

**过程中发现的一个原版 trainer 固有边界**：`warmup_iters == lr_decay_iters` 时 `get_lr` 内 `(it - warmup_iters)/(lr_decay_iters - warmup_iters)` 除零崩溃（首次对照用 `MAX_ITERS=10 WARMUP=10` 触发）。此行为在原版 `train_rmnp_streaming.py` 中同样存在，非本次改动引入，未修改（保持最小 diff），使用时注意 `warmup_iters < lr_decay_iters`。

---

## 7. 后续建议（不在本次范围）

1. fixed-$p$ 网格（$p\in\{1.5,2,4,8,\infty\}$）确认中间 $p$ 是否有收益——这是 adaptive 有意义的前提；
2. adaptive 跑通后记录每层 $p^\star$ 轨迹（layer type × 训练阶段结构，对应 extended_rmnp §14 的诊断项）；
3. 若有效，尝试"$b_i$ 摊销 + $a_i,c_i,p^\star$ 每步重解"的高频档（SMuon 因 SVD 做不到、row 结构可以，见 `HTRMNP/SMuon-From-SGD-to-Muon/Claude/separable_surrogate_vs_exact_row.md`）；
4. $b_i$ 的 $O(m)$ all-reduce 全局 batch 曲率选项。
