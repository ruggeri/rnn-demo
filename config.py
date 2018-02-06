# Training
BATCH_SIZE = 101
BATCH_STRING_LENGTH = 64
FNAME = "21839.txt"
LEARNING_RATE = 1e-3
NUM_CHARS = 128
NUM_EPOCHS = 100
LAYER1_SIZE = 256
LAYER2_SIZE = 257

# Generation
BURN_IN_CHARS = 2 ** 13
CHARS_TO_GENERATE = 2 ** 11

# Got ~30% accuracy with no hidden layer.
# Got ~50% accuracy with two layers of 32/128/129
# Not clear that 32/512/513 is better...
