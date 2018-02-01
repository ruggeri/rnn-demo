from config import *
from keras.utils import to_categorical
import numpy as np

text = None
with open(FNAME, 'r') as f:
    text = f.read()

def split_text(text):
    int_text = [
        to_categorical(ord(c), 256)
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
        batch = [
            subtext[(batch_idx * BATCH_STRING_LENGTH):((batch_idx + 1) * BATCH_STRING_LENGTH)]
            for subtext in subtexts
        ]

        batches.append(
            np.array(batch)
        )

    return batches

batches = split_text(text)
