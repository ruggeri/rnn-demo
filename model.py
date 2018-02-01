from config import *
import numpy as np
import tensorflow as tf

def build_graph(string_length, train_mode):
    # Placeholders
    start_character = tf.placeholder(
        dtype = tf.float64,
        shape = (None, NUM_CHARS),
        name = "start_character",
    )
    target_characters = tf.placeholder(
        dtype = tf.float64,
        shape = (None, string_length, NUM_CHARS),
        name = "target_characters",
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
    initial_layer3_state = tf.placeholder(
        dtype = tf.float64,
        shape = (None, LAYER3_SIZE),
        name = "layer3_initial_state",
    )

    # Weights
    emission_matrix = tf.Variable(
        np.random.normal(
            size = (LAYER3_SIZE, NUM_CHARS),
            scale = np.sqrt(1 / (LAYER3_SIZE + NUM_CHARS))
        ),
        name = "emission_matrix",
    )
    emission_bias = tf.Variable(
        np.zeros((NUM_CHARS,)),
        name = "emission_bias"
    )

    layer3_transition_matrix = tf.Variable(
        np.random.normal(
            size = (LAYER2_SIZE + LAYER3_SIZE, LAYER3_SIZE),
            scale = np.sqrt(1 / (LAYER2_SIZE + LAYER3_SIZE + LAYER3_SIZE)),
        ),
        name = "layer3_transition_matrix",
    )
    layer3_transition_bias = tf.Variable(
        np.zeros((LAYER3_SIZE,)),
        name = "layer3_transition_bias",
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
    current_layer3_state = initial_layer3_state
    prev_emission = start_character
    for string_idx in range(string_length):
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

        layer3_ipt = tf.concat(
            [current_layer3_state, current_layer2_state],
            axis = 1
        )

        current_layer3_state = tf.matmul(
            layer3_ipt,
            layer3_transition_matrix
        ) + layer3_transition_bias
        current_layer3_state = tf.nn.relu(current_layer3_state)

        current_emission_logits = tf.matmul(
            current_layer3_state, emission_matrix
        ) + emission_bias
        current_emission_probs = tf.nn.softmax(
            current_emission_logits,
        )

        all_emission_logits.append(current_emission_logits)
        all_emission_probs.append(current_emission_probs)

        # Teacher forcing.
        prev_emission = target_characters[:, string_idx, :]

    final_layer1_state = current_layer1_state
    final_layer2_state = current_layer2_state
    final_layer3_state = current_layer3_state

    # Calculate loss
    total_loss = 0.0
    accuracies = []
    for string_idx in range(string_length):
        current_emission_logits = all_emission_logits[string_idx]
        predicted_emission = tf.argmax(current_emission_logits, axis = 1)

        correct_emission = tf.argmax(
            target_characters[:, string_idx, :],
            axis = 1
        )

        total_loss += tf.nn.softmax_cross_entropy_with_logits_v2(
            labels = target_characters[:, string_idx, :],
            logits = current_emission_logits
        ) / string_length

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

    if train_mode:
        optimizer = tf.train.AdamOptimizer(
            learning_rate = LEARNING_RATE
        )

        #gradients, variables = zip(*optimizer.compute_gradients(mean_loss))
        #gradients, _ = tf.clip_by_global_norm(gradients, GRADIENT_CLIP)
        #train_step = optimizer.apply_gradients(zip(gradients, variables))
        train_step = optimizer.minimize(mean_loss)
    else:
        train_step = None

    graph = {
        "start_character": start_character,
        "target_characters": target_characters,
        "initial_layer1_state": initial_layer1_state,
        "initial_layer2_state": initial_layer2_state,
        "initial_layer3_state": initial_layer3_state,

        "final_layer1_state": final_layer1_state,
        "final_layer2_state": final_layer2_state,
        "final_layer3_state": final_layer3_state,
        "final_emission_probs": all_emission_probs[-1],

        "mean_loss": mean_loss,
        "accuracy": accuracy,
        "train_step": train_step,
    }

    return graph
