from config import *
import numpy as np
import tensorflow as tf

def build_graph(string_length, train_mode):
    # Placeholders

    # The family of Dashwood
    target_characters = tf.placeholder(
        tf.float64,
        (None, BATCH_STRING_LENGTH, NUM_CHARS),
        name = "target_characters"
    )
    # 256 zeros for the initial state.
    initial_layer1_state = tf.placeholder(
        tf.float64,
        (None, LAYER1_SIZE),
        name = "initial_layer1_state"
    )
    initial_layer2_state = tf.placeholder(
        tf.float64,
        (None, LAYER2_SIZE),
        name = "initial_layer2_state"
    )

    # Character zero because no prior character ever written.
    start_character = tf.placeholder(
        tf.float64,
        (None, NUM_CHARS),
        name = "start_character"
    )

    layer1_transition_matrix = tf.Variable(
        np.random.normal(
            size = (LAYER1_SIZE + NUM_CHARS, LAYER1_SIZE),
            scale = np.sqrt(1 / (LAYER1_SIZE + NUM_CHARS + LAYER1_SIZE))
        ),
        name = "layer1_transition_matrix"
    )
    layer1_biases = tf.Variable(
        np.zeros(LAYER1_SIZE),
        name = "layer1_biases"
    )

    layer2_transition_matrix = tf.Variable(
        np.random.normal(
            size = (LAYER2_SIZE + LAYER1_SIZE, LAYER2_SIZE),
            scale = np.sqrt(1 / (LAYER2_SIZE + LAYER1_SIZE + LAYER2_SIZE))
        ),
        name = "layer2_transition_matrix"
    )
    layer2_biases = tf.Variable(
        np.zeros(LAYER2_SIZE),
        name = "layer2_biases"
    )

    emissions_matrix = tf.Variable(
        np.random.normal(
            size = (LAYER2_SIZE, NUM_CHARS),
            scale = np.sqrt(1 / (LAYER2_SIZE + NUM_CHARS))
        ),
        name = "emissions_matrix"
    )
    emissions_biases = tf.Variable(
        np.zeros(NUM_CHARS),
        # dtype = tf.float64,
        name = "emissions_biases"
    )

    current_layer1_state = initial_layer1_state
    current_layer2_state = initial_layer2_state
    current_character = start_character
    all_emission_probability_logits = []
    all_emission_probabilities = []
    for string_idx in range(BATCH_STRING_LENGTH):
        ipt1 = tf.concat([
            current_layer1_state,
            current_character,
        ], axis = 1)
        next_layer1_state = tf.matmul(
            ipt1,
            layer1_transition_matrix
        ) + layer1_biases
        next_layer1_state = tf.nn.relu(
            next_layer1_state
        )

        ipt2 = tf.concat([
            current_layer2_state,
            next_layer1_state,
        ], axis = 1)
        next_layer2_state = tf.matmul(
            ipt2,
            layer2_transition_matrix
        ) + layer2_biases
        next_layer2_state = tf.nn.relu(
            next_layer2_state
        )

        # vector of size NUM_CHARS, and the range of values?
        # -inf to +inf
        next_emission_probability_logits = tf.matmul(
            next_layer2_state,
            emissions_matrix
        ) + emissions_biases
        all_emission_probability_logits.append(
            next_emission_probability_logits
        )
        # vector of size NUM_CHARS, values are zero to one.
        # vector must sum to one, because it's a prob distribution.
        next_emission_probabilities = tf.nn.softmax(
            next_emission_probability_logits
        )
        all_emission_probabilities.append(next_emission_probabilities)

        # teacher forcing
        current_layer1_state = next_layer1_state
        current_layer2_state = next_layer2_state
        current_character = target_characters[:, string_idx, :]

    # Compute losses:
    all_losses = []
    all_accuracies = []
    for string_idx, logits in enumerate(all_emission_probability_logits):
        correct_answer = target_characters[:, string_idx, :]
        correct_answer_code = tf.argmax(
            correct_answer,
            axis = 1
        )
        guessed_letter_code = tf.argmax(
            logits,
            axis = 1
        )
        current_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels = correct_answer,
                logits = logits
            )
        )
        current_accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(correct_answer_code, guessed_letter_code),
                tf.float64
            )
        )
        all_losses.append(current_loss)
        all_accuracies.append(current_accuracy)

    total_mean_loss = tf.reduce_mean(all_losses)
    total_accuracy = tf.reduce_mean(all_accuracies)

    train_step = tf.train.AdamOptimizer(
        learning_rate = LEARNING_RATE
    ).minimize(total_mean_loss)

    final_layer1_state = current_layer1_state
    final_layer2_state = current_layer2_state

    return {
        "target_characters": target_characters,
        "initial_layer1_state": initial_layer1_state,
        "initial_layer2_state": initial_layer2_state,
        "start_character": start_character,

        "total_mean_loss": total_mean_loss,
        "total_accuracy": total_accuracy,
        "train_step": train_step,
        "final_layer1_state": final_layer1_state,
        "final_layer2_state": final_layer2_state
    }
