#!/usr/bin/env pythonw

from __future__ import print_function
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.optimizers import RMSprop
import numpy as np
import json

import re
from collections import Counter

import urllib.request

import matplotlib.pyplot as plt

filename = "Frankenstein.txt"

word_regex = re.compile(r'\w+', re.U)

text = "".join(open(filename, 'rt').readlines())

text_words = word_regex.findall(text)

word_counts = Counter(text_words)

word_count_items = list(word_counts.items())
word_count_items.sort(key=lambda x: x[1], reverse=True)

word_list = [word for word, count in word_count_items]
word_dict = dict(zip(word_list, range(len(word_count_items))))

number_words = len(word_counts)

print("Creating input and label text...")
sequence_length = 10
step = 1

input_words = []
label_words = []
for i in range(0, len(text_words) - sequence_length, step):
    input_words.append(text_words[i:i + sequence_length])
    label_words.append(text_words[i + sequence_length])

print('Vectorization...')

X = np.zeros((len(input_words), sequence_length, number_words), dtype=np.bool)
y = np.zeros((len(input_words), number_words), dtype=np.bool)

for i, sentence in enumerate(input_words):
    for t, word in enumerate(sentence):
        X[i, t, word_dict[word]] = 1

    y[i, word_dict[label_words[i]]] = 1

HIDDEN_SIZE = 128
BATCH_SIZE = 128
NUM_ITERATIONS = 1
NUM_EPOCHS_PER_ITERATION = 30
NUM_PREDS_PER_EPOCH = 0

# build the model: a single LSTM
print('Build model...')
model_simple = Sequential()
model_simple.add(LSTM(128, input_shape=(sequence_length, number_words)))
model_simple.add(Dense(number_words))
model_simple.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model_simple.compile(loss='categorical_crossentropy', optimizer=optimizer)

def train_model(model, X_train, y_train, val_split=0.1):
    for iteration in range(NUM_ITERATIONS):
        print("=" * 50)
        print("Iteration #: %d" % (iteration))
        
        # We keep the history result of the trainig step. This provides us with 
        # valueable information about the training procedure that we may use later.
        history = model.fit(X_train, y_train, 
                            validation_split=val_split,
                            batch_size=BATCH_SIZE, epochs=NUM_EPOCHS_PER_ITERATION)

        # Choose a specific sentence to use to generate the predictions 
        # and generate NUM_PREDS_PER_EPOCH predictions.
        test_idx = 1502 
        test_words = input_words[test_idx]
        print("Generating from seed: \"%s\"" % (" ".join(test_words)))

        for i in range(NUM_PREDS_PER_EPOCH):
            Xtest = np.zeros((1, sequence_length, number_words))

            for i, word in enumerate(test_words):
                Xtest[0, i, word_dict[word]] = 1

            pred = model.predict(Xtest, verbose=0)[0]
            ypred = word_list[np.argmax(pred)]
            print(" ".join(test_words), "=>", ypred, end=" ")

            # Add the new word to the test instance and 
            # move the window forward 1 step
            test_words = test_words[1:] + [ypred]

            print()
            
    return history, model

history, model = train_model(model_simple, X, y, 0.1)

history_dict = history.history

model_filename = "lstm_simple_01"

model.save(model_filename + ".h5")
json.dump(history_dict, open(model_filename + ".json", "w"))

loss = history_dict['loss']
validation_loss = history_dict['val_loss']

epochs = list(range(1, NUM_EPOCHS_PER_ITERATION+1))

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, validation_loss, 'r', label = 'Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(model_filename + ".png", dpi=300)
#plt.show()
