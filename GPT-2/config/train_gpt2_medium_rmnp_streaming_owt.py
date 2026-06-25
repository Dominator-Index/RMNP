wandb_log = True
wandb_project = 'mars-owt'
wandb_run_name='gpt2-medium-rmnp-streaming-20k'

batch_size = 15
block_size = 1024
gradient_accumulation_steps = 4

n_layer = 24
n_head = 16
n_embd = 1024
dropout = 0.0
bias = False
scale_attn_by_inverse_layer_idx = True

max_iters = 20000
lr_decay_iters = 20000

eval_interval = 1000
eval_iters = 200
log_interval = 10

# optimizer
optimizer_name = 'rmnp'
learning_rate = 1.5e-3
weight_decay = 1e-1
rmnp_learning_rate = 5e-3
rmnp_weight_decay = 0.
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 2000
min_lr = 6e-5
schedule = 'cosine'
compile = True

# Streaming configuration
use_streaming = True
streaming_timeout = 7200
streaming_max_retries = 10
streaming_dataset = "Skylion007/openwebtext"

out_dir = 'out_medium_rmnp_streaming_20k'
