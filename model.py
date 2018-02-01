from config import *
import numpy as np
import tensorflow as tf

# Placeholders
input_characters = tf.placeholder(
    dtype = tf.float64,
    shape = (None, BATCH_STRING_LENGTH, NUM_CHARS),
    name = "input_characters",
)

initial_layer1_state = tf.placeholder(
    dtype = tf.float64,
    shape = (None, LAYER1_SIZE),
    name = "layer1_initial_state",
)
initial_layer2_state = tf.placeholder(
    dtype = tf.float64,
    shape = (None, LAYER2_SIZE),
    name = "layer2_initial_state",
)

# Weights
emission_matrix = tf.Variable(
    np.random.normal(
        size = (LAYER2_SIZE, NUM_CHARS),
        scale = np.sqrt(1 / (LAYER2_SIZE + NUM_CHARS))
    ),
    name = "emission_matrix",
)
emission_bias = tf.Variable(
    np.zeros((NUM_CHARS,)),
    name = "emission_bias"
)

layer2_transition_matrix = tf.Variable(
    np.random.normal(
        size = (LAYER1_SIZE + LAYER2_SIZE, LAYER2_SIZE),
        scale = np.sqrt(1 / (LAYER1_SIZE + LAYER2_SIZE + LAYER2_SIZE)),
    ),
    name = "layer2_transition_matrix",
)
layer2_transition_bias = tf.Variable(
    np.zeros((LAYER2_SIZE,)),
    name = "layer2_transition_bias",
)

layer1_to_layer2_matrix = tf.Variable(
    np.random.normal(
        size = (LAYER1_SIZE, LAYER2_SIZE),
        scale = np.sqrt(1 / (LAYER1_SIZE + LAYER2_SIZE))
    ),
    name = "layer1_to_layer2_matrix",
)
layer1_to_layer2_bias = tf.Variable(
    np.zeros((LAYER2_SIZE,)),
    name = "layer1_to_layer2_biases",
)

layer1_transition_matrix = tf.Variable(
    np.random.normal(
        size = (LAYER1_SIZE + NUM_CHARS, LAYER1_SIZE),
        scale = np.sqrt(1 / (LAYER1_SIZE + NUM_CHARS + LAYER1_SIZE)),
    ),
    name = "layer1_transition_matrix",
)
layer1_transition_bias = tf.Variable(
    np.zeros((LAYER1_SIZE,)),
    name = "layer1_transition_bias",
)

# Build emissions sequence.
all_emission_logits = []
all_emission_probs = []

current_layer1_state = initial_layer1_state
current_layer2_state = initial_layer2_state
prev_emission = input_characters[:, 0, :]
for string_idx in range(BATCH_STRING_LENGTH - 1):
    layer1_ipt = tf.concat(
        [current_layer1_state, prev_emission],
        axis = 1
    )

    current_layer1_state = tf.matmul(
        layer1_ipt,
        layer1_transition_matrix,
    ) + layer1_transition_bias
    current_layer1_state = tf.nn.relu(current_layer1_state)

    layer2_ipt = tf.concat(
        [current_layer2_state, current_layer1_state],
        axis = 1
    )

    current_layer2_state = tf.matmul(
        layer2_ipt,
        layer2_transition_matrix
    ) + layer2_transition_bias
    current_layer2_state = tf.nn.relu(current_layer2_state)

    current_emission_logits = tf.matmul(
        current_layer2_state, emission_matrix
    ) + emission_bias
    current_emission_probs = tf.nn.softmax(
        current_emission_logits,
    )
    all_emission_logits.append(current_emission_logits)
    all_emission_probs.append(current_emission_probs)

    # Teacher forcing.
    prev_emission = input_characters[:, string_idx + 1, :]

final_layer1_state = current_layer1_state
final_layer2_state = current_layer2_state

# Calculate loss
total_loss = 0.0
accuracies = []
for string_idx in range(BATCH_STRING_LENGTH - 1):
    current_emission_logits = all_emission_logits[string_idx]
    predicted_emission = tf.argmax(current_emission_logits, axis = 1)

    correct_emission = tf.argmax(
        input_characters[:, string_idx + 1, :],
        axis = 1
    )

    total_loss += tf.nn.softmax_cross_entropy_with_logits_v2(
        labels = input_characters[:, string_idx + 1, :],
        logits = current_emission_logits
    ) / (BATCH_STRING_LENGTH - 1)

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

train_step = tf.train.AdamOptimizer(
    learning_rate = LEARNING_RATE
).minimize(mean_loss)

graph = {
    "input_characters": input_characters,
    "initial_layer1_state": initial_layer1_state,
    "initial_layer2_state": initial_layer2_state,

    "final_layer1_state": final_layer1_state,
    "final_layer2_state": final_layer2_state,
    "all_emission_probs": all_emission_probs,

    "mean_loss": mean_loss,
    "accuracy": accuracy,
    "train_step": train_step,
}
