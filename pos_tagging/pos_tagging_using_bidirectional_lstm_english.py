# -*- coding: utf-8 -*-
"""POS Tagging Using Bidirectional LSTM English.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19mB5x40-3g7rGmRFxX99h8H6XL2FS6F9
"""

from google.colab import drive  
drive.mount('/gdrive')
import warnings
warnings.filterwarnings("ignore")

import numpy as np

from matplotlib import pyplot as plt

import re
import seaborn as sns

from gensim.models import KeyedVectors

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense, Input
from keras.layers import TimeDistributed
from keras.layers import LSTM, GRU, Bidirectional, SimpleRNN, RNN
from keras.models import Model
from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

data_path = "/gdrive/MyDrive/2nd Semester/Coursework/NLP/data/"
training_tags = data_path + "ptb.2-21.tgs"
training_txt = data_path + "ptb.2-21.txt"
validation_data = data_path + "ptb.22.txt"

words, tags = set([]), set([])
with open(training_tags) as tag_file, open(training_txt) as token_file:
    for tagString, tokenString in zip(tag_file, token_file):
        tag = re.split("\s+", tagString.rstrip())
        token = re.split("\s+", tokenString.rstrip())
        pairs = list(zip(tag, token))
        for (tag, token) in pairs:
          words.add(token.lower())
          tags.add(tag.lower())



word2index = {w: i + 2 for i, w in enumerate(list(words))}
word2index['-PAD-'] = 0  # The special value used for padding
word2index['-OOV-'] = 1  # The special value used for OOVs
 
tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
tag2index['-PAD-'] = 0  # The special value used to padding

sentences = []
sentence_tags = []

with open(training_tags) as tag_file, open(training_txt) as token_file:
    for tagString, tokenString in zip(tag_file, token_file):
      sentence_tag = re.split("\s+", tagString.rstrip())
      sentence_token = re.split("\s+", tokenString.rstrip())
      sentences.append(sentence_token)
      sentence_tags.append(sentence_tag)

from sklearn.model_selection import train_test_split
train_sentences, test_sentences, train_tags, test_tags = train_test_split(sentences, sentence_tags, test_size=0.2)

words, tags = set([]), set([])
 
for s in train_sentences:
    for w in s:
        words.add(w.lower())
 
for ts in train_tags:
    for t in ts:
        tags.add(t)
 
word2index = {w: i + 2 for i, w in enumerate(list(words))}
word2index['-PAD-'] = 0  # The special value used for padding
word2index['-OOV-'] = 1  # The special value used for OOVs

tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
tag2index['-PAD-'] = 0  # The special value used to padding

path = '/gdrive/MyDrive/2nd Semester/Coursework/NLP/GoogleNews-vectors-negative300.bin'
word2vec = KeyedVectors.load_word2vec_format(path, binary=True)
EMBEDDING_SIZE  = 300  # each word in word2vec model is represented using a 300 dimensional vector
VOCABULARY_SIZE = len(word2index)
embedding_weights = np.zeros((VOCABULARY_SIZE, EMBEDDING_SIZE))


for word, index in word2index.items():
    try:
        embedding_weights[index, :] = word2vec[word]
    except KeyError:
        pass



train_sentences_X, test_sentences_X, train_tags_y, test_tags_y = [], [], [], []
 
for s in train_sentences:
    s_int = []
    for w in s:
        try:
            s_int.append(word2index[w.lower()])
        except KeyError:
            s_int.append(word2index['-OOV-'])
 
    train_sentences_X.append(s_int)
 
for s in test_sentences:
    s_int = []
    for w in s:
        try:
            s_int.append(word2index[w.lower()])
        except KeyError:
            s_int.append(word2index['-OOV-'])
 
    test_sentences_X.append(s_int)
 
for s in train_tags:
    train_tags_y.append([tag2index[t] for t in s])
 
for s in test_tags:
    test_tags_y.append([tag2index[t] for t in s])
 
MAX_LENGTH = len(max(train_sentences_X, key=len))

from keras.preprocessing.sequence import pad_sequences
 
train_sentences_X = pad_sequences(train_sentences_X, maxlen=MAX_LENGTH, padding='post')
test_sentences_X = pad_sequences(test_sentences_X, maxlen=MAX_LENGTH, padding='post')
train_tags_y = pad_sequences(train_tags_y, maxlen=MAX_LENGTH, padding='post')
test_tags_y = pad_sequences(test_tags_y, maxlen=MAX_LENGTH, padding='post')

from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from tensorflow.keras.optimizers import Adam
 
 
model = Sequential()
model.add(InputLayer(input_shape=(MAX_LENGTH, )))
model.add(Embedding(len(word2index), 300, weights = [embedding_weights],trainable = True))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
# model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(TimeDistributed(Dense(len(tag2index))))
model.add(Activation('softmax'))
 
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.001),
              metrics=['accuracy'])
 
model.summary()

def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)

def logits_to_tokens(sequences, index):
    token_sequences = []
    for categorical_sequence in sequences:
        token_sequence = []
        for categorical in categorical_sequence:
            token_sequence.append(index[np.argmax(categorical)])
 
        token_sequences.append(token_sequence)
 
    return token_sequences

cat_train_tags_y = to_categorical(train_tags_y, len(tag2index))
print(cat_train_tags_y[0])

model.fit(train_sentences_X, to_categorical(train_tags_y, len(tag2index)), batch_size=128, epochs=10, validation_split=0.2)

scores = model.evaluate(test_sentences_X, to_categorical(test_tags_y, len(tag2index)))
print(f"{model.metrics_names[1]}: {scores[1] * 100}")   # acc: 99.09751977804825

test_samples = []
with open(validation_data) as VALIDATION_DATA:
  for line in VALIDATION_DATA:
    tokens = re.split("\s+", line.rstrip())
    test_samples.append(tokens)

test_samples_X = []
for s in test_samples:
    s_int = []
    for w in s:
        try:
            s_int.append(word2index[w.lower()])
        except KeyError:
            s_int.append(-1)
    test_samples_X.append(s_int)
 
test_samples_X = pad_sequences(test_samples_X, maxlen=MAX_LENGTH, padding='post')
print(test_samples_X[0])

predictions = model.predict(test_samples_X)

pred = logits_to_tokens(predictions, {i: t for t, i in tag2index.items()})

truncated_prediction_sequence = []
with open(validation_data) as VALIDATION_DATA:
  for index, line in enumerate(VALIDATION_DATA):
    tokens = re.split("\s+", line.rstrip())
    truncated_prediction_sequence.append(pred[index][:len(tokens)])

import os
from pathlib import Path
root = "/gdrive/MyDrive/2nd Semester/Coursework/NLP/"
results = os.path.join(root, 'english_bidirectional_lstm.out')
# Path(results).mkdir(parents=True, exist_ok=True)
print(results)
f = open(results, "w+")
for sequence in truncated_prediction_sequence:
  f.write(" ".join(sequence) + "\n")
print("Writing done")

GOLD_FILE = "/gdrive/MyDrive/2nd Semester/Coursework/NLP/ptb.22.tgs"
HYPO_FILE = results

def evalaute_tag_acc(golds, hypos):
    tag_errors = 0
    sent_errors = 0
    tag_tot = 0
    sent_tot = 0

    for g, h in zip(golds, hypos):
        g = g.strip()
        h = h.strip()

        g_toks = re.split("\s+", g)
        h_toks = re.split("\s+", h)

        error_flag = False

        for i in range(len(g_toks)):
            if i >= len(h_toks) or g_toks[i] != h_toks[i]:
                tag_errors += 1
                error_flag = True

            tag_tot += 1

        if error_flag:
            sent_errors += 1

        sent_tot += 1

    print("error rate by word:      ", tag_errors / tag_tot, f" ({tag_errors} errors out of {tag_tot})")
    print("error rate by sentence:  ", sent_errors / sent_tot, f" ({sent_errors} errors out of {sent_tot})")


with open(GOLD_FILE) as goldFile, open(HYPO_FILE) as hypoFile:
    golds = goldFile.readlines()
    hypos = hypoFile.readlines()
    evalaute_tag_acc(golds, hypos)
