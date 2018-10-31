import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
import jsonlines

word_counts_set = set()
input_file = jsonlines.open('/Users/jdo/dev/data/word_learning/data/sanitized_list_3.jsonl', 'r')
count_file = jsonlines.open('/Users/jdo/dev/data/word_learning/scripts/word_counts.jsonl', 'r')


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
model.add(Embedding(vocab_size, 10, input_length=max_length - 1))
model.add(LSTM(50))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# compile network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(X, y, epochs=10, verbose=2)

model.save('garms_nlp_model_1.h5')
# evaluate model
print(generate_seq(model, tokenizer, max_length - 1, 'dress', 4))
print(generate_seq(model, tokenizer, max_length - 1, 'shirt', 4))
