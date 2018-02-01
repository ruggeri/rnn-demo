from config import *
from dataset import text
from model import build_graph
from keras.utils import to_categorical
import numpy as np
import tensorflow as tf

session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

graph = build_graph(string_length = 1, train_mode = False)

saver = tf.train.Saver()
saver.restore(session, './models/model.ckpt-7')

current_layer1_state = np.random.uniform(
    size = (1, LAYER1_SIZE),
)
current_layer2_state = np.random.uniform(
    size = (1, LAYER2_SIZE),
)


def predict_next_char(start_char):
    global current_layer1_state, current_layer2_state

    one_hot_start_char = to_categorical(
        ord(start_char),
        NUM_CHARS
    )
    one_hot_start_char = np.expand_dims(
        one_hot_start_char,
        axis = 0
    )

    emission_probs, curent_layer1_state, current_layer2_state = session.run(
        [graph["final_emission_probs"],
         graph["final_layer1_state"],
         graph["final_layer2_state"],
        ],
        feed_dict = {
            graph["start_character"]: one_hot_start_char,
            graph["initial_layer1_state"]: current_layer1_state,
            graph["initial_layer2_state"]: current_layer2_state,
        }
    )
    emission_probs = np.squeeze(emission_probs)

    return chr(np.random.choice(
        np.arange(NUM_CHARS),
        p = emission_probs
    ))

for idx in range(BURN_IN_CHARS):
    prev_char = predict_next_char(text[idx])

string = ""
for _ in range(CHARS_TO_GENERATE):
    prev_char = predict_next_char(prev_char)
    string += prev_char

# This is required to handle codepoints outside the range.
bstring = string.encode('ascii', errors = 'ignore')
string = bstring.decode('ascii')
print(string)
