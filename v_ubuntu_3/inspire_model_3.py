import numpy as np
import random
import pickle
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, BatchNormalization
from keras.layers import LSTM, Bidirectional
from keras.layers import Embedding
# from keras.layers import TimeDistributed
from keras.layers import Dropout
import jsonlines

word_counts_set = set()
input_file = jsonlines.open('sanitized_list_4.jsonl', 'r')
count_file = jsonlines.open('word_counts.jsonl', 'r')


# generate a sequence from a language model
def generate_seq(seq_model, seq_tokenizer, max_len, seed_text, n_words):
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded_seq = seq_tokenizer.texts_to_sequences([in_text])[0]
        # pre-pad sequences to a fixed length
        encoded_seq = pad_sequences([encoded_seq], maxlen=max_len, padding='pre')
        # predict probabilities for each word
        yhat = seq_model.predict_classes(encoded_seq, verbose=0)
        # map predicted word index to word
        out_word = ''
        for word, index in seq_tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
    return in_text


def load_text(word_file):
    text = ''

    count_counter = 0
    for count_line in count_file:
        # print(count_line[1])
        if int(count_line[1]) > 49:
            # word_counts[count_line[0]] = int(count_line[1])
            count_counter += 1
            word_counts_set.add(count_line[0])
            print(f'count lines in: {count_counter}')

    for input_line in word_file:
        randomizer = random.random()
        if randomizer > 0.6:
            write_array = [x for x in input_line if x in word_counts_set]
            write_line = ' '.join(write_array) + '\n'
            text += write_line

    return text


data = load_text(input_file)

# prepare the tokenizer on the source text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
# determine the vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

# saving tokenizer
with open('tokenizer_10.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# create line-based sequences
sequences = list()
for line in data.split('\n'):
    encoded = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(encoded)):
        sequence = encoded[:i + 1]
        sequences.append(sequence)
print('Total Sequences: %d' % len(sequences))
# pad input sequences
max_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
print('Max Sequence Length: %d' % max_length)
# split into input and output elements
sequences = np.array(sequences)
X, y = sequences[:, :-1], sequences[:, -1]
y = to_categorical(y, num_classes=vocab_size)
# define model
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=max_length - 1))

model.add(Bidirectional(LSTM(
    300,
    kernel_initializer='glorot_uniform',
    recurrent_initializer='orthogonal',
    recurrent_activation='sigmoid',
    bias_initializer='zeros',
    unit_forget_bias=True,
    kernel_regularizer=None,
    recurrent_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    recurrent_constraint=None,
    bias_constraint=None,
    return_sequences=True,
    return_state=False,
    stateful=False
)))

model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Bidirectional(LSTM(
    300,
    kernel_initializer='glorot_uniform',
    recurrent_initializer='orthogonal',
    recurrent_activation='sigmoid',
    bias_initializer='zeros',
    unit_forget_bias=True,
    kernel_regularizer=None,
    recurrent_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    recurrent_constraint=None,
    bias_constraint=None,
    return_sequences=True,
    return_state=False,
    stateful=False
)))

model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Bidirectional(LSTM(
    300,
    kernel_initializer='glorot_uniform',
    recurrent_initializer='orthogonal',
    recurrent_activation='sigmoid',
    bias_initializer='zeros',
    unit_forget_bias=True,
    kernel_regularizer=None,
    recurrent_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    recurrent_constraint=None,
    bias_constraint=None,
    return_sequences=False,
    return_state=False,
    stateful=False
)))

model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(vocab_size))
model.add(Activation('softmax'))
print(model.summary())
# compile network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(X, y, epochs=150, batch_size=1048)

# # saving tokenizer
# with open('tokenizer_10.pickle', 'wb') as handle:
#     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

model.save('garms_nlp_model_10.h5')
# evaluate model
print(generate_seq(model, tokenizer, max_length - 1, 'red', 4))
print(generate_seq(model, tokenizer, max_length - 1, 'lace', 4))
print(generate_seq(model, tokenizer, max_length - 1, 'mini', 4))
print(generate_seq(model, tokenizer, max_length - 1, 'push', 4))
print(generate_seq(model, tokenizer, max_length - 1, 'bra', 4))
print(generate_seq(model, tokenizer, max_length - 1, 'black', 4))
print(generate_seq(model, tokenizer, max_length - 1, 'striped', 4))
print(generate_seq(model, tokenizer, max_length - 1, 'floral', 4))
print(generate_seq(model, tokenizer, max_length - 1, 'polka', 4))
