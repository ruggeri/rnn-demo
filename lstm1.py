from config import *
import numpy as np
import tensorflow as tf

forget_gate1_weights = tf.Variable(
    np.random.normal(
        size = (LAYER1_SIZE + NUM_CHARS, LAYER1_SIZE),
        scale = np.sqrt(1 / (LAYER1_SIZE + NUM_CHARS + LAYER1_SIZE))
    ),
    name = "forget_gate1_weights"
)
forget_gate1_biases = tf.Variable(
    np.ones((LAYER1_SIZE,)),
    name = "forget_gate1_biases"
)

write_gate1_weights = tf.Variable(
    np.random.normal(
        size = (LAYER1_SIZE + NUM_CHARS, LAYER1_SIZE),
        scale = np.sqrt(1 / (LAYER1_SIZE + NUM_CHARS + LAYER1_SIZE))
    ),
    name = "write_gate1_weights"
)
write_gate1_biases = tf.Variable(
    np.zeros((LAYER1_SIZE,)),
    name = "write_gate1_biases"
)

update_gate1_weights = tf.Variable(
    np.random.normal(
        size = (LAYER1_SIZE + NUM_CHARS, LAYER1_SIZE),
        scale = np.sqrt(1 / (LAYER1_SIZE + NUM_CHARS + LAYER1_SIZE))
    ),
    name = "update_gate1_weights"
)
update_gate1_biases = tf.Variable(
    np.zeros((LAYER1_SIZE,)),
    name = "update_gate1_biases"
)

output_gate1_weights = tf.Variable(
    np.random.normal(
        size = (LAYER1_SIZE + NUM_CHARS, LAYER1_SIZE),
        scale = np.sqrt(1 / (LAYER1_SIZE + NUM_CHARS + LAYER1_SIZE))
    ),
    name = "output_gate1_weights"
)
output_gate1_biases = tf.Variable(
    np.zeros((LAYER1_SIZE,)),
    name = "output_gate1_biases"
)

def lstm_layer1(prev_state, prev_output, current_input):
    gate_inputs = tf.concat([
        prev_output,
        current_input
    ], axis = 1)

    forget_gate_output = tf.matmul(
        gate_inputs,
        forget_gate1_weights
    ) + forget_gate1_biases
    forget_gate_output = tf.nn.sigmoid(forget_gate_output)

    next_state = prev_state * forget_gate_output

    write_gate_output = tf.matmul(
        gate_inputs,
        write_gate1_weights
    ) + write_gate1_biases
    write_gate_output = tf.nn.sigmoid(write_gate_output)

    update_values = tf.matmul(
        gate_inputs,
        update_gate1_weights
    ) + update_gate1_biases
    update_values = tf.nn.tanh(update_values)

    next_state = next_state + (write_gate_output * update_values)

    output_gate_output = tf.matmul(
        gate_inputs,
        output_gate1_weights
    ) + output_gate1_biases
    output_gate_output = tf.nn.sigmoid(output_gate_output)

    next_output = tf.nn.tanh(next_state) * output_gate_output

    return (next_state, next_output)
