# Original params
# BLOCK_SIZE = 16
# NUM_BLOCKS = 3
# N_EMB = 32
# N_HEAD = 4
# HEAD_SIZE = N_EMB//N_HEAD
# DROP_RATE = 0.2
# BATCH_SIZE = 32
# NUM_EPOCHS = 1000
# LEARNING_RATE = 1e-3

# See Karpathy 1:40:00 for suggested big hyperparams. (5-01 checkpoints)
# BATCH_SIZE = 64
# NUM_EPOCHS = 5000
# LEARNING_RATE = 3e-4
# BLOCK_SIZE = 256
# NUM_BLOCKS = 6
# N_EMB = 384
# N_HEAD = 6
# HEAD_SIZE = N_EMB//N_HEAD
# DROP_RATE = 0.2

# WORD PARTS v1 (5-03 checkpoints)
# Vocab size is now dominating the params, so might as well scale up a bit!
# karpathy = 10M
# karpathy + word parts = 50M
# **this + word parts = 77M
# BATCH_SIZE = 16 #64 caused CUDA out of memory
# NUM_EPOCHS = 5000
# LEARNING_RATE = 3e-5
# BLOCK_SIZE = 256 # about 4x the context, since word parts
# NUM_BLOCKS = 8 # double blocks --> 60M
# N_EMB = 512 # 60 -> 90 (1024 yields 250M! Yikes.) 
# N_HEAD = 8
# HEAD_SIZE = N_EMB//N_HEAD
# DROP_RATE = 0.2

# WORD PARTS v2 (5-04 checkpoints)
# (GPT2 117M, https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) 
# This: 163M
BATCH_SIZE = 4 #even just 8 caused CUDA out of memory error on the 15GB free GPUs!
NUM_EPOCHS = 5000
LEARNING_RATE = 3e-7 #trying pretty small LR because my batch size is so small.
BLOCK_SIZE = 1024
NUM_BLOCKS = 12
N_EMB = 768
N_HEAD = 12
HEAD_SIZE = N_EMB//N_HEAD
DROP_RATE = 0.2

# GPT3 Small (see few shot learner paper) 
# NUM_EPOCHS = 600k?? # get through all 300B tokens in data
# LEARNING_RATE = 6e-4
# BLOCK_SIZE = 2048 # subword tokens, not characters
# BATCH_SIZE = 500k//BLOCK_SIZE # 250?? (I think this is total token, thus divide)
# NUM_BLOCKS = 12
# N_EMB = 768
# N_HEAD = 12
# HEAD_SIZE = N_EMB//N_HEAD
# DROP_RATE = 0.2


# Full GPT3
# NUM_EPOCHS = 100k?? # get through all 300B tokens in data
# LEARNING_RATE = 6e-5
# BLOCK_SIZE = 2048 # subword tokens, not characters
# BATCH_SIZE = 3.2M//BLOCK_SIZE # 1500? (I think this is total token, thus divide)
# NUM_BLOCKS = 96
# N_EMB = 12288
# N_HEAD = 96
# HEAD_SIZE = N_EMB//N_HEAD
# DROP_RATE = 0.2