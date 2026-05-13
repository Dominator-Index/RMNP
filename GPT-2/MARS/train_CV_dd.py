"""
CV Training with Diagonal Dominance Monitoring on ResNet-18 + CIFAR-10/100.
Computes Gram matrix diagonal dominance metrics (u, ratio_to_max, ratio_to_avg)
for each 2D parameter's gradient, logged to wandb per epoch.
"""
import numpy as np
import os
import argparse
import json
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch CV Training with DD Monitoring')
parser.add_argument("--dataset", type=str, default="cifar10", choices=["mnist", "cifar10", "cifar100"])
parser.add_argument("--scheduler", type=str, default="multistep", choices=["multistep", "cosine", "constant"])
parser.add_argument("--train_bsz", type=int, default=128)
parser.add_argument("--eval_bsz", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--cpu", action="store_true")
parser.add_argument("--cuda", type=str, default="0")
parser.add_argument('--lr', default=0.02, type=float, help='learning rate (matrix lr for muon/rnnps)')
parser.add_argument('--adamw_lr', default=0.003, type=float, help='learning rate for adamw (1D params)')
parser.add_argument('--resume', '-r', action='store_true')
parser.add_argument('--optim', '-m', type=str, choices=["adam", "adamw", "muon", "rnnps"], default='muon')
parser.add_argument('--net', '-n', type=str, default="resnet18")
parser.add_argument('--wd', default=0., type=float, help='weight decay')
parser.add_argument('--Nepoch', default=200, type=int)
parser.add_argument('--beta1', default=0.9, type=float)
parser.add_argument('--beta2', default=0.999, type=float)
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--save_dir', type=str, default="./checkpoint")
parser.add_argument('--wandb_name', type=str, default="None")
parser.add_argument('--wandb_project', type=str, default="CV-DD", help='wandb project name')
parser.add_argument('--dd_every', type=int, default=10, help='compute DD metrics every N batches (0=disabled)')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
if args.wandb:
    import wandb
    if args.wandb_name == "None":
        wandb.init(project=args.wandb_project, name=f"{args.dataset}_{args.net}_{args.optim}_lr{args.lr}_dd", config=args)
    else:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=args)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from utils.cv_utils import get_datasets, get_scheduler, get_model

use_cuda = torch.cuda.is_available() and not args.cpu

os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


def compute_diagonal_dominance(g):
    """
    Compute diagonal dominance metrics for Gram matrix g @ g^T.
    g: tensor of shape (m, n), flattened to 2D if needed.
    """
    if g.ndim > 2:
        g = g.reshape(g.size(0), -1)
    if g.ndim < 2 or g.size(0) < 2:
        return None

    gram = g.float() @ g.float().T
    m = gram.shape[0]

    diag = torch.diag(gram)
    off_diag_sum = torch.sum(torch.abs(gram), dim=1) - torch.abs(diag)
    u = diag - off_diag_sum

    gram_abs = torch.abs(gram)
    mask = torch.eye(m, dtype=torch.bool, device=gram.device)
    gram_for_max = gram_abs.clone()
    gram_for_max[mask] = float('-inf')
    off_diag_max_per_row = gram_for_max.max(dim=1).values
    off_diag_avg_per_row = (gram_abs.sum(dim=1) - diag.abs()) / (m - 1 + 1e-10)

    ratio_to_max = diag / (off_diag_max_per_row + 1e-10)
    ratio_to_avg = diag / (off_diag_avg_per_row + 1e-10)

    return {
        "u_avg": u.mean().item(), "u_min": u.min().item(), "u_max": u.max().item(),
        "ratio_to_max_avg": ratio_to_max.mean().item(),
        "ratio_to_max_min": ratio_to_max.min().item(),
        "ratio_to_max_max": ratio_to_max.max().item(),
        "ratio_to_avg_avg": ratio_to_avg.mean().item(),
        "ratio_to_avg_min": ratio_to_avg.min().item(),
        "ratio_to_avg_max": ratio_to_avg.max().item(),
    }


def compute_all_dd_metrics(model):
    """Compute DD metrics for all 2D+ parameters with gradients."""
    dd_log = {}
    all_ratio_to_avg_avg = []
    all_ratio_to_avg_min = []

    for name, p in model.named_parameters():
        if p.grad is None or p.ndim < 2:
            continue
        grad = p.grad.data
        if grad.ndim > 2:
            grad = grad.reshape(grad.size(0), -1)
        metrics = compute_diagonal_dominance(grad)
        if metrics is None:
            continue

        # Per-layer metrics (all 9)
        clean_name = name.replace("module.", "").replace(".", "/")
        dd_log[f"dd_u/{clean_name}/u_avg"] = metrics["u_avg"]
        dd_log[f"dd_u/{clean_name}/u_min"] = metrics["u_min"]
        dd_log[f"dd_u/{clean_name}/u_max"] = metrics["u_max"]
        dd_log[f"dd_ratio/{clean_name}/ratio_to_max_avg"] = metrics["ratio_to_max_avg"]
        dd_log[f"dd_ratio/{clean_name}/ratio_to_max_min"] = metrics["ratio_to_max_min"]
        dd_log[f"dd_ratio/{clean_name}/ratio_to_max_max"] = metrics["ratio_to_max_max"]
        dd_log[f"dd_ratio/{clean_name}/ratio_to_avg_avg"] = metrics["ratio_to_avg_avg"]
        dd_log[f"dd_ratio/{clean_name}/ratio_to_avg_min"] = metrics["ratio_to_avg_min"]
        dd_log[f"dd_ratio/{clean_name}/ratio_to_avg_max"] = metrics["ratio_to_avg_max"]

        all_ratio_to_avg_avg.append(metrics["ratio_to_avg_avg"])
        all_ratio_to_avg_min.append(metrics["ratio_to_avg_min"])

    # Global averages
    if all_ratio_to_avg_avg:
        dd_log["dd_ratio/global/ratio_to_avg_avg"] = sum(all_ratio_to_avg_avg) / len(all_ratio_to_avg_avg)
        dd_log["dd_ratio/global/ratio_to_avg_min"] = sum(all_ratio_to_avg_min) / len(all_ratio_to_avg_min)

    return dd_log


# Build model and data
trainloader, testloader = get_datasets(args.dataset, args.train_bsz, args.eval_bsz)
print('==> Building model..')
model = get_model(args)

if use_cuda:
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
betas = (args.beta1, args.beta2)

# Optimizer setup
from optimizers.muon import Muon
from optimizers.rnnps import RNNPS
from opt import CombinedOptimizer
from optimizers.adamw import AdamW

if args.optim == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=betas)
elif args.optim == 'adamw':
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=betas)
elif args.optim == 'muon':
    optimizer = CombinedOptimizer(model.parameters(), [AdamW, Muon],
                                  [{'lr': args.adamw_lr, 'betas': betas, 'weight_decay': args.wd},
                                   {'lr': args.lr, 'weight_decay': 0.}])
elif args.optim == 'rnnps':
    optimizer = CombinedOptimizer(model.parameters(), [AdamW, RNNPS],
                                  [{'lr': args.adamw_lr, 'betas': betas, 'weight_decay': args.wd},
                                   {'lr': args.lr, 'weight_decay': 0.}])

scheduler = get_scheduler(optimizer, args)
best_acc = 0
global_step = 0

# Print parameter grouping info
n_2d = sum(1 for n, p in model.named_parameters() if p.ndim >= 2)
n_1d = sum(1 for n, p in model.named_parameters() if p.ndim < 2)
print(f"Parameters: {n_2d} 2D+ (DD monitored), {n_1d} 1D")

t_bar = tqdm(total=len(trainloader))
t_bar2 = tqdm(total=len(testloader))

for epoch in range(1, args.Nepoch + 1):
    scheduler.step()
    model.train()

    train_loss = 0
    correct_train = 0
    total_train = 0
    t_bar.reset()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        global_step += 1
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        # Compute DD metrics after optimizer step
        dd_log = {}
        if args.dd_every > 0 and global_step % args.dd_every == 0:
            if args.optim == 'muon':
                # Read DD from Muon's internal computation (after Nesterov momentum, before NS)
                # This matches the GPT-2 baseline (train_muon_streaming_test.py)
                muon_opt = optimizer.optimizers[1]  # CombinedOptimizer: [AdamW, Muon]
                for pname, metrics in muon_opt.dd_per_param.items():
                    dd_log[f"dd_u/{pname}/u_avg"] = metrics["u_avg"]
                    dd_log[f"dd_u/{pname}/u_min"] = metrics["u_min"]
                    dd_log[f"dd_u/{pname}/u_max"] = metrics["u_max"]
                    dd_log[f"dd_ratio/{pname}/ratio_to_max_avg"] = metrics["ratio_to_max_avg"]
                    dd_log[f"dd_ratio/{pname}/ratio_to_max_min"] = metrics["ratio_to_max_min"]
                    dd_log[f"dd_ratio/{pname}/ratio_to_max_max"] = metrics["ratio_to_max_max"]
                    dd_log[f"dd_ratio/{pname}/ratio_to_avg_avg"] = metrics["ratio_to_avg_avg"]
                    dd_log[f"dd_ratio/{pname}/ratio_to_avg_min"] = metrics["ratio_to_avg_min"]
                    dd_log[f"dd_ratio/{pname}/ratio_to_avg_max"] = metrics["ratio_to_avg_max"]
                dd_log["dd_u/global/u_bar_avg"] = muon_opt.dd_u_bar_avg
                dd_log["dd_u/global/u_bar_min"] = muon_opt.dd_u_bar_min
                dd_log["dd_u/global/u_bar_max"] = muon_opt.dd_u_bar_max
                # Global ratio averages (6 metrics, matching train_muon_streaming_test.py)
                all_ratio_to_max_avg = [m["ratio_to_max_avg"] for m in muon_opt.dd_per_param.values()]
                all_ratio_to_max_min = [m["ratio_to_max_min"] for m in muon_opt.dd_per_param.values()]
                all_ratio_to_max_max = [m["ratio_to_max_max"] for m in muon_opt.dd_per_param.values()]
                all_ratio_to_avg_avg = [m["ratio_to_avg_avg"] for m in muon_opt.dd_per_param.values()]
                all_ratio_to_avg_min = [m["ratio_to_avg_min"] for m in muon_opt.dd_per_param.values()]
                all_ratio_to_avg_max = [m["ratio_to_avg_max"] for m in muon_opt.dd_per_param.values()]
                if all_ratio_to_max_avg:
                    n = len(all_ratio_to_max_avg)
                    dd_log["dd_ratio/global/ratio_to_max_avg"] = sum(all_ratio_to_max_avg) / n
                    dd_log["dd_ratio/global/ratio_to_max_min"] = sum(all_ratio_to_max_min) / n
                    dd_log["dd_ratio/global/ratio_to_max_max"] = sum(all_ratio_to_max_max) / n
                    dd_log["dd_ratio/global/ratio_to_avg_avg"] = sum(all_ratio_to_avg_avg) / n
                    dd_log["dd_ratio/global/ratio_to_avg_min"] = sum(all_ratio_to_avg_min) / n
                    dd_log["dd_ratio/global/ratio_to_avg_max"] = sum(all_ratio_to_avg_max) / n
            else:
                # For RNNPS/others: compute DD on raw gradients (no internal DD available)
                dd_log = compute_all_dd_metrics(model)

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += targets.size(0)
        correct_train += predicted.eq(targets.data).cpu().sum().item()

        t_bar.update(1)
        t_bar.set_description('Epoch: %d | Loss: %.3f | Acc: %.3f%% ' %
                              (epoch, train_loss / (batch_idx + 1), 100.0 * correct_train / total_train))
        t_bar.refresh()

        # Log DD metrics
        if args.wandb and dd_log:
            dd_log["global_step"] = global_step
            wandb.log(dd_log, step=global_step)

    # Epoch-level logging
    train_acc = 100.0 * correct_train / total_train

    # Eval
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    t_bar2.reset()

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()
        t_bar2.update(1)
        t_bar2.set_description('Loss: %.3f | Acc: %.3f%% (Best: %.3f%%)' %
                               (test_loss / (batch_idx + 1), 100.0 * correct / total, best_acc))
        t_bar2.refresh()

    test_acc = 100.0 * correct / total
    if args.wandb:
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss / (batch_idx + 1),
            "train_acc": train_acc,
            "test_loss": test_loss / (batch_idx + 1),
            "test_acc": test_acc,
            "lr": scheduler.get_lr()[0],
        }, step=global_step)

    if test_acc > best_acc:
        best_acc = test_acc

print(f"\nBest test accuracy: {best_acc:.2f}%")
