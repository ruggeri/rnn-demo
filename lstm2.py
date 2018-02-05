from config import *
import numpy as np
import tensorflow as tf

forget_gate2_weights = tf.Variable(
    np.random.normal(
        size = (LAYER2_SIZE + LAYER1_SIZE, LAYER2_SIZE),
        scale = np.sqrt(1 / (LAYER2_SIZE + LAYER1_SIZE + LAYER2_SIZE))
    ),
    name = "forget_gate2_weights"
)
forget_gate2_biases = tf.Variable(
    np.ones((LAYER2_SIZE,)),
    name = "forget_gate2_biases"
)

write_gate2_weights = tf.Variable(
    np.random.normal(
        size = (LAYER2_SIZE + LAYER1_SIZE, LAYER2_SIZE),
        scale = np.sqrt(1 / (LAYER2_SIZE + LAYER1_SIZE + LAYER2_SIZE))
    ),
    name = "write_gate2_weights"
)
write_gate2_biases = tf.Variable(
    np.zeros((LAYER2_SIZE,)),
    name = "write_gate2_biases"
)

update_gate2_weights = tf.Variable(
    np.random.normal(
        size = (LAYER2_SIZE + LAYER1_SIZE, LAYER2_SIZE),
        scale = np.sqrt(1 / (LAYER2_SIZE + LAYER1_SIZE + LAYER2_SIZE))
    ),
    name = "update_gate2_weights"
)
update_gate2_biases = tf.Variable(
    np.zeros((LAYER2_SIZE,)),
    name = "update_gate2_biases"
)

output_gate2_weights = tf.Variable(
    np.random.normal(
        size = (LAYER2_SIZE + LAYER1_SIZE, LAYER2_SIZE),
        scale = np.sqrt(1 / (LAYER2_SIZE + LAYER1_SIZE + LAYER2_SIZE))
    ),
    name = "output_gate2_weights"
)
output_gate2_biases = tf.Variable(
    np.zeros((LAYER2_SIZE,)),
    name = "output_gate2_biases"
)

def lstm_layer2(prev_state, prev_output, current_input):
    gate_inputs = tf.concat([
        prev_output,
        current_input
    ], axis = 1)

    forget_gate_output = tf.matmul(
        gate_inputs,
        forget_gate2_weights
    ) + forget_gate2_biases
    forget_gate_output = tf.nn.sigmoid(forget_gate_output)

    next_state = prev_state * forget_gate_output

    write_gate_output = tf.matmul(
        gate_inputs,
        write_gate2_weights
    ) + write_gate2_biases
    write_gate_output = tf.nn.sigmoid(write_gate_output)

    update_values = tf.matmul(
        gate_inputs,
        update_gate2_weights
    ) + update_gate2_biases
    update_values = tf.nn.tanh(update_values)

    next_state = next_state + (write_gate_output * update_values)

    output_gate_output = tf.matmul(
        gate_inputs,
        output_gate2_weights
    ) + output_gate2_biases
    output_gate_output = tf.nn.sigmoid(output_gate_output)

    next_output = tf.nn.tanh(next_state) * output_gate_output

    return (next_state, next_output)
