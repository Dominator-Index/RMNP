wandb_log = True
wandb_project = 'rmnp'
wandb_run_name='gpt2-small-rmnp-streaming-10k'

batch_size = 15
block_size = 1024
gradient_accumulation_steps = 4

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

max_iters = 10000
lr_decay_iters = 10000

eval_interval = 1000
eval_iters = 200
log_interval = 10

# optimizer
optimizer_name = 'rmnp'
learning_rate = 3e-3
weight_decay = 1e-1
rmnp_learning_rate = 4e-3
rmnp_weight_decay = 0.
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 1000
min_lr = 3e-5
schedule = 'cosine'
compile = True

# Streaming configuration
use_streaming = True
streaming_timeout = 7200
streaming_max_retries = 10
streaming_dataset = "Skylion007/openwebtext"

out_dir = 'out_small_rmnp_streaming_10k'
