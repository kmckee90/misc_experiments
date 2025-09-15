# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = "out-shakespeare"
eval_interval = 250  # keep frequent because we'll overfit
eval_iters = 100
log_interval = 10  # don't print too too often

init_from = "resume"

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True

wandb_log = False  # override via command line if you like
wandb_project = "shakespeare"
wandb_run_name = "mini-gpt"

dataset = "shakespeare"
gradient_accumulation_steps = 4
batch_size = 256
block_size = 64  # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 240
dropout = 0.2

learning_rate = 1e-3  # with baby networks can afford to go a bit higher
max_iters = 3500
lr_decay_iters = 3500  # make equal to max_iters usually
min_lr = 1e-3  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 100  # not super necessary potentially

compile = False
# on macbook also add
device = "cuda:0"  # run on cpu only
# compile = False # do not torch compile the model
