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

def run_batch(current_layer1_state, start_character, batch):
    _, final_layer1_state, total_mean_loss, total_accuracy = session.run(
        [graph["train_step"],
         graph["final_layer1_state"],
         graph["total_mean_loss"],
         graph["total_accuracy"],
        ], feed_dict = {
            graph["start_character"]: start_character,
            graph["target_characters"]: batch,
            graph["initial_state"]: current_layer1_state,
        }
    )

    return (final_layer1_state, total_mean_loss, total_accuracy)

def run_epoch(epoch_idx):
    current_layer1_state = np.zeros((BATCH_SIZE, LAYER1_SIZE))

    start_character = np.zeros((BATCH_SIZE, NUM_CHARS))
    for batch_idx, batch in enumerate(batches):
        current_layer1_state, total_mean_loss, total_accuracy = run_batch(
            current_layer1_state,
            start_character,
            batch,
        )

        start_character = batch[:, -1, :]

        print(
            f'E {epoch_idx:04d} | '
            f'B {batch_idx:04d} | '
            f'L {total_mean_loss:0.2f} | '
            f'A {total_accuracy:0.2f}'
        )

saver = tf.train.Saver()
saver.save(session, './models/model.ckpt', global_step = 0)
for epoch_idx in range(NUM_EPOCHS):
    run_epoch(epoch_idx)
    saver.save(session, './models/model.ckpt', global_step = epoch_idx)
