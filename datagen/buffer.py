import random
from collections import deque


class Buffer:
    def __init__(self, maxlen=None):
        self.batches = deque(maxlen=maxlen)

    def add(self, tokens):
        self.batches.append(tokens.detach())

    def sample(self, n, curr_batch=None):
        k = min(n, len(self.batches))
        indices = random.sample(range(len(self.batches)), k)
        batches_list = list(self.batches)
        result_tokens = [batches_list[i] for i in indices]
        if curr_batch is not None:
            result_tokens.append(curr_batch)
        return result_tokens

    def __len__(self):
        return len(self.batches)
