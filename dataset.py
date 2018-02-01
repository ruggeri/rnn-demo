from config import *
import numpy as np

def to_categorical(idx):
    assert idx < 256
    assert 0 <= idx
    res = np.zeros(NUM_CHARS)
    res[idx] = 1.0
    return res

text = None
with open(FNAME, 'r') as f:
    text = f.read()

def split_text(text):
    int_text = [
        to_categorical(ord(c))
        for c in text
    ]

    subtexts = []
    subtext_length = len(int_text) // BATCH_SIZE
    subtexts = [
        int_text[(idx * subtext_length):((idx + 1) * subtext_length)]
        for idx in range(BATCH_SIZE)
    ]

    num_batches = subtext_length // BATCH_STRING_LENGTH
    batches = []
    for batch_idx in range(num_batches):
        batch_start = batch_idx * BATCH_STRING_LENGTH
        batch_end = (batch_idx + 1) * BATCH_STRING_LENGTH
        batch = [subtext[batch_start:batch_end] for subtext in subtexts]
        batches.append(batch)

    return np.array(batches)

batches = split_text(text)
print(batches.shape)
