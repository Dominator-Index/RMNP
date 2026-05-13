wandb_log = True
wandb_project = 'mars-fw'
wandb_run_name='gpt2-xl-adamw-streaming-fw-50k'

batch_size = 120
block_size = 1024
gradient_accumulation_steps = 1

n_layer = 48
n_head = 25
n_embd = 1600
dropout = 0.0
bias = False
scale_attn_by_inverse_layer_idx = True

max_iters = 50000
lr_decay_iters = 50000

eval_interval = 1000
eval_iters = 200
log_interval = 10

# optimizer
optimizer_name = 'adamw'
learning_rate = 2e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 5000
min_lr = 1e-5
schedule = 'cosine'
compile = True

# Streaming configuration
use_streaming = True
streaming_timeout = 7200
streaming_max_retries = 10
streaming_dataset = "karpathy/fineweb-edu-100b-shuffle"

out_dir = 'out_xl_adamw_streaming_fw_50k'
