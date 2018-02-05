from config import *
from dataset import batches
from model import build_graph
import numpy as np
import tensorflow as tf

graph = build_graph(
    string_length = BATCH_STRING_LENGTH,
    train_mode = True
)

session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

def run_batch(prior_state, start_character, batch):
    keys = {
        "initial_layer1_state",
        "initial_layer1_output",
        "initial_layer2_state",
        "initial_layer2_output",
    }
    state_input = {
        graph[key]: prior_state[key] for key in keys
    }
    state_input[graph["start_character"]] = start_character
    state_input[graph["target_characters"]] = batch

    _, next_state, metrics = session.run(
        [
            graph["train_step"],
            {
                "initial_layer1_state": graph["final_layer1_state"],
                "initial_layer1_output": graph["final_layer1_output"],
                "initial_layer2_state": graph["final_layer2_state"],
                "initial_layer2_output": graph["final_layer2_output"],
            },
            {
                "total_mean_loss": graph["total_mean_loss"],
                "total_accuracy": graph["total_accuracy"],
            }
        ], feed_dict = state_input
    )

    return next_state, metrics

def run_epoch(epoch_idx):
    current_state = {
        "initial_layer1_state": np.zeros((BATCH_SIZE, LAYER1_SIZE)),
        "initial_layer1_output": np.zeros((BATCH_SIZE, LAYER1_SIZE)),
        "initial_layer2_state": np.zeros((BATCH_SIZE, LAYER2_SIZE)),
        "initial_layer2_output": np.zeros((BATCH_SIZE, LAYER2_SIZE)),
    }

    start_character = np.zeros((BATCH_SIZE, NUM_CHARS))
    for batch_idx, batch in enumerate(batches):
        current_state, metrics = run_batch(
            current_state,
            start_character,
            batch,
        )

        start_character = batch[:, -1, :]

        print(
            f'E {epoch_idx:04d} | '
            f'B {batch_idx:04d} | '
            f'L {metrics["total_mean_loss"]:0.2f} | '
            f'A {metrics["total_accuracy"]:0.2f}'
        )

saver = tf.train.Saver()
saver.save(session, './models/model.ckpt', global_step = 0)
for epoch_idx in range(NUM_EPOCHS):
    run_epoch(epoch_idx)
    saver.save(session, './models/model.ckpt', global_step = epoch_idx)
