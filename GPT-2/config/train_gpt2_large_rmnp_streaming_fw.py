wandb_log = True
wandb_project = 'mars-fw'
wandb_run_name='gpt2-large-rmnp-streaming-fw-100k'

batch_size = 15
block_size = 1024
gradient_accumulation_steps = 4

n_layer = 36
n_head = 20
n_embd = 1280
dropout = 0.0
bias = False
scale_attn_by_inverse_layer_idx = True

max_iters = 40000
lr_decay_iters = 40000

eval_interval = 1000
eval_iters = 200
log_interval = 10

# optimizer
optimizer_name = 'rmnp'
learning_rate = 1e-3
weight_decay = 1e-1
rmnp_learning_rate = 3e-3
rmnp_weight_decay = 0.
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 4000
min_lr = 1e-5
schedule = 'cosine'
compile = True

# Streaming configuration
use_streaming = True
streaming_timeout = 7200
streaming_max_retries = 10
streaming_dataset = "karpathy/fineweb-edu-100b-shuffle"

out_dir = 'out_large_rmnp_streaming_fw_100k'
