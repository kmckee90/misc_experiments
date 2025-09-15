# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = "out-owt-res-big"
eval_interval = 250  # keep frequent because we'll overfit
eval_iters = 50
log_interval = 10  # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True

wandb_log = False  # override via command line if you like
wandb_project = "owt"
wandb_run_name = "mini-gpt"

# init_from = 'scratch'
init_from = "resume"

dataset = "openwebtext"
gradient_accumulation_steps = 16
batch_size = 64
block_size = 512  # context of up to 256 previous characters

# baby GPT model :)
n_layer = 2
n_head = 6
n_embd = 1024
dropout = 0.2

learning_rate = 1e-4  # with baby networks can afford to go a bit higher
max_iters = 20000
lr_decay_iters = 20000  # make equal to max_iters usually
min_lr = 1e-5  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 100  # not super necessary potentially

compile = False
# on macbook also add
device = "cuda:3"  # run on cpu only
# compile = False # do not torch compile the model
