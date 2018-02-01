from config import *
from dataset import batches
from model import graph
import numpy as np
import tensorflow as tf

session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

def run_batch(current_state, batch):
    _, final_state, mean_loss, accuracy = session.run(
        [graph["train_step"],
         graph["final_state"],
         graph["mean_loss"],
         graph["accuracy"],
        ], feed_dict = {
            graph["input_characters"]: batch,
            graph["initial_state"]: current_state,
        }
    )

    return (final_state, mean_loss, accuracy)

def run_epoch(epoch_idx):
    current_state = np.random.uniform(
        size = (BATCH_SIZE, STATE_SIZE),
    )

    for batch_idx, batch in enumerate(batches):
        final_state, mean_loss, accuracy = run_batch(
            current_state,
            batch,
        )

        current_state = final_state

        print(
            f'E {epoch_idx:04d} | '
            f'B {batch_idx:04d} | '
            f'L {mean_loss:0.2f} | '
            f'A {accuracy:0.2f}'
        )

for epoch_idx in range(NUM_EPOCHS):
    run_epoch(epoch_idx)
