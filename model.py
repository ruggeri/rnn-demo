import numpy as np
import tensorflow as tf

STRING_LENGTH = 32
STATE_SIZE = 128
NUM_CHARS = 256

# Placeholders
input_characters = tf.placeholder(
    dtype = tf.float64,
    shape = (None, STRING_LENGTH, NUM_CHARS),
    name = "input_characters",
)

initial_state = tf.placeholder(
    dtype = tf.float64,
    shape = (None, STATE_SIZE),
    name = "initial_state",
)

# Weights
emission_matrix = tf.Variable(
    np.random.normal(
        size = (STATE_SIZE, NUM_CHARS),
        scale = np.sqrt(1 / (STATE_SIZE + NUM_CHARS))
    ),
    name = "emission_matrix",
)
emission_bias = tf.Variable(
    np.zeros((NUM_CHARS,)),
    name = "emission_bias"
)

transition_matrix = tf.Variable(
    np.random.normal(
        size = (STATE_SIZE + NUM_CHARS, STATE_SIZE),
        scale = np.sqrt(1 / (STATE_SIZE + STATE_SIZE))
    ),
    name = "transition_matrix",
)
transition_bias = tf.Variable(
    np.zeros((STATE_SIZE,)),
    name = "transition_bias",
)

# Build emissions sequence.
all_emission_logits = []
all_emission_probs = []
current_state = initial_state
prev_emission = input_characters[:, 0, :]
for string_idx in range(STRING_LENGTH - 1):
    current_emission_logits = tf.matmul(
        current_state, emission_matrix
    ) + emission_bias
    current_emission_probs = tf.nn.softmax(
        current_emission_logits,
    )
    all_emission_logits.append(current_emission_logits)
    all_emission_probs.append(current_emission_probs)

    ipt = tf.concat(
        [current_state, prev_emission],
        axis = 1
    )

    current_state = tf.matmul(
        ipt,
        transition_matrix
    ) + transition_bias

    # Teacher forcing.
    prev_emission = input_characters[:, string_idx + 1, :]

# Calculate loss
total_loss = 0.0
accuracies = []
for string_idx in range(STRING_LENGTH - 1):
    current_emission_logits = all_emission_logits[string_idx]
    predicted_emission = tf.cast(
        tf.argmax(
            current_emission_logits,
            axis = 1,
        ),
        tf.float64
    )
    correct_emission = input_characters[:, string_idx + 1, :]

    total_loss += tf.nn.softmax_cross_entropy_with_logits_v2(
        labels = input_characters[:, string_idx, :],
        logits = current_emission_logits
    )

    accuracy = tf.reduce_mean(
        tf.cast(
            tf.equal(
                predicted_emission,
                correct_emission
            ),
            tf.float64
        )
    )
    accuracies.append(accuracy)

mean_loss = tf.reduce_mean(total_loss)
accuracy = tf.reduce_mean(accuracies)

graph = {
    "input_characters": input_characters,
    "initial_state": initial_state,
    "mean_loss": mean_loss,
    "accuracy": accuracy,
    "all_emission_probs": all_emission_probs
}
