import numpy as np
import torch
import random

# random seeds
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

a = np.arange(10)
batch_size = 5
seq_len = 3
limit = 10 - seq_len

idx = np.random.randint(0, limit, size=batch_size)

samples = a[idx : idx + seq_len]

print(samples)
