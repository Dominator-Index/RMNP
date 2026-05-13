wandb_log = True
wandb_project = 'rmnp'
wandb_run_name='gpt2-large-adamw-streaming-100k'

batch_size = 5
block_size = 1024
gradient_accumulation_steps = 24

n_layer = 36
n_head = 20
n_embd = 1280
dropout = 0.0
bias = False
scale_attn_by_inverse_layer_idx = True

max_iters = 100000
lr_decay_iters = 100000

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
warmup_iters = 10000
min_lr = 1e-5
schedule = 'cosine'
compile = True

# Streaming configuration
use_streaming = True
streaming_timeout = 7200
streaming_max_retries = 10
# Dataset options: "karpathy/fineweb-edu-100b-shuffle" or "Skylion007/openwebtext"
streaming_dataset = "Skylion007/openwebtext"

out_dir = 'out_large_adamw_streaming_100k'
