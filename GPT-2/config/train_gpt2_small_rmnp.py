wandb_log = True
wandb_project = 'rmnp'
wandb_run_name='gpt2-small-rmnp-10k'

batch_size = 15
block_size = 1024
gradient_accumulation_steps = 4

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False

# this makes total number of tokens be ~50B
max_iters = 10000
lr_decay_iters = 10000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# optimizer
optimizer_name = 'rmnp'
learning_rate = 3e-3 # max learning rate, original=6e-4
weight_decay = 1e-1
rmnp_learning_rate = 4e-3
rmnp_weight_decay = 0.
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 1000 # how many steps to warm up for
min_lr = 3e-5
schedule = 'cosine'
compile = True

out_dir = 'out_small_rmnp_10k'
