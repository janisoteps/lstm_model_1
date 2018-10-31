import jsonlines
# import random
from keras.models import load_model
# from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle


word_counts_set = set()
input_file = jsonlines.open('/Users/jdo/dev/data/garms_nlp/lstm_model/v_ubuntu_3/sanitized_list_4.jsonl', 'r')
count_file = jsonlines.open('/Users/jdo/dev/data/garms_nlp/data/word_counts.jsonl', 'r')


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


# def load_text(word_file):
#     text = ''
#
#     count_counter = 0
#     for count_line in count_file:
#         # print(count_line[1])
#         if int(count_line[1]) > 49:
#             # word_counts[count_line[0]] = int(count_line[1])
#             count_counter += 1
#             word_counts_set.add(count_line[0])
#             print(f'count lines in: {count_counter}')
#
#     for input_line in word_file:
#         randomizer = random.random()
#         if randomizer > 0.6:
#             write_array = [x for x in input_line if x in word_counts_set]
#             write_line = ' '.join(write_array) + '\n'
#             text += write_line
#
#     return text
#
#
# data = load_text(input_file)
#
# # prepare the tokenizer on the source text
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts([data])

# loading
# with open('tokenizer.pickle', 'rb') as handle:
handle = open('/Users/jdo/dev/data/garms_nlp/lstm_model/v_ubuntu_3/tokenizer_10.pickle', 'rb')
tokenizer = pickle.load(handle)

inspire_model = load_model('/Users/jdo/dev/data/garms_nlp/lstm_model/v_ubuntu_3/garms_nlp_model_10.h5')

inspire_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(generate_seq(inspire_model, tokenizer, 27 - 1, 'red shoe', 1))
print(generate_seq(inspire_model, tokenizer, 27 - 1, 'lace dress', 1))
print(generate_seq(inspire_model, tokenizer, 27 - 1, 'mini dress', 1))
print(generate_seq(inspire_model, tokenizer, 27 - 1, 'push up', 1))
print(generate_seq(inspire_model, tokenizer, 27 - 1, 'black dress', 1))
print(generate_seq(inspire_model, tokenizer, 27 - 1, 'swimsuit with', 1))
print(generate_seq(inspire_model, tokenizer, 27 - 1, 'floral trousers', 1))
print('\n')
print(generate_seq(inspire_model, tokenizer, 27 - 1, 'red shoe', 2))
print(generate_seq(inspire_model, tokenizer, 27 - 1, 'lace dress', 2))
print(generate_seq(inspire_model, tokenizer, 27 - 1, 'mini dress', 2))
print(generate_seq(inspire_model, tokenizer, 27 - 1, 'push up', 2))
print(generate_seq(inspire_model, tokenizer, 27 - 1, 'black dress', 2))
print(generate_seq(inspire_model, tokenizer, 27 - 1, 'swimsuit with', 2))
print(generate_seq(inspire_model, tokenizer, 27 - 1, 'floral trousers', 2))
print('\n')
print(generate_seq(inspire_model, tokenizer, 27 - 1, 'red shoe', 3))
print(generate_seq(inspire_model, tokenizer, 27 - 1, 'lace dress', 3))
print(generate_seq(inspire_model, tokenizer, 27 - 1, 'mini dress', 3))
print(generate_seq(inspire_model, tokenizer, 27 - 1, 'push up', 3))
print(generate_seq(inspire_model, tokenizer, 27 - 1, 'black dress', 3))
print(generate_seq(inspire_model, tokenizer, 27 - 1, 'swimsuit with', 3))
print(generate_seq(inspire_model, tokenizer, 27 - 1, 'floral trousers', 3))
print('\n')
print(generate_seq(inspire_model, tokenizer, 27 - 1, 'red', 3))
print(generate_seq(inspire_model, tokenizer, 27 - 1, 'laced', 3))
print(generate_seq(inspire_model, tokenizer, 27 - 1, 'mini', 3))
print(generate_seq(inspire_model, tokenizer, 27 - 1, 'push', 3))
print(generate_seq(inspire_model, tokenizer, 27 - 1, 'dress', 3))
print(generate_seq(inspire_model, tokenizer, 27 - 1, 'swimming', 3))
print(generate_seq(inspire_model, tokenizer, 27 - 1, 'floral', 3))
