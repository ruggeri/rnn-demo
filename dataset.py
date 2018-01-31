from keras.utils import to_categorical

BATCH_SIZE = 32
NUM_CHARS = 256
STRING_LENGTH = 32

def split_text(text):
    int_text = [
        to_categorical(ord(c), 256)
        for c in text
    ]

    subtexts = []
    subtext_length = len(int_text) // BATCH_SIZE
    subtexts = [
        int_text[(idx * subtext_length):((idx + 1) * subtext_length)]
        for idx in range(subtext_length)
    ]

    num_batches = subtext_length // STRING_LENGTH
    batches = []
    for batch_idx in range(num_batches):
        batch = [
            subtext[(batch_idx * STRING_LENGTH):((batch_idx + 1) * STRING_LENGTH)]
            for subtext in subtexts
        ]

        batches.append(
            np.array(batch)
        )

    return batches
