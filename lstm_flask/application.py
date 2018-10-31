from flask import Flask, jsonify, request
import jsonlines
import random
import json
import pickle
# import nltk
from keras.models import load_model
# from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
sess = tf.Session()


sim_word_file = jsonlines.open('/Users/jdo/dev/data/garms_nlp/data/word_suggestions_3.jsonl')
sim_word_dict = {}  # Structure: key: [[word,sim,count],...]
for sim_word in sim_word_file:
    sim_word_dict[sim_word['word']] = sim_word['similar_words']


# # # # Flask Itself # # # #
application = app = Flask(__name__)


# generate a sequence from a language model
def generate_seq(seq_model, seq_tokenizer, max_len, seed_text, n_words):
    in_text = seed_text
    # generate a fixed number of words
    words = seed_text.split(' ')
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
        words.append(out_word)

    print(f'words: {words}')
    uniq_words = uniq_list(words)
    output = ' '.join(uniq_words)

    return output, uniq_words


def uniq_list(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


# # # # API Functions # # # #
@app.route("/api/sequences", methods=["POST"])
def features():
    if request.method == "POST":
        req_data = request.get_json(force=True)
        print(req_data)
        print(req_data['input_text'])
        input_text = req_data['input_text'].split(' ')
        sanitized_words = [x.strip(':{}()+=_*&^<>%$£#@!`"\\\'?[]') for x in input_text]
        input_word_count = len(sanitized_words)
        sanitized_string = ' '.join(sanitized_words)
        print(f'Sanitized string: {sanitized_string}')

        with graph.as_default():
            if input_word_count == 1:
                main_pred, main_pred_list = generate_seq(inspire_model, tokenizer, 27 - 1, sanitized_string, 4)
                update_word = main_pred_list[-4]
            else:
                main_pred, main_pred_list = generate_seq(inspire_model, tokenizer, 27 - 1, sanitized_string, 3)
                update_word = main_pred_list[-3]

            similar_words = sim_word_dict[update_word]
            top_3_sim_words = [x[0] for x in similar_words[:3]]
            print(f'top 3 sim words: {top_3_sim_words}')
            if input_word_count == 1:
                sim_suggestion_list = [' '.join(main_pred_list[:-4]) + ' ' + x for x in top_3_sim_words]
            else:
                sim_suggestion_list = [' '.join(main_pred_list[:-3]) + ' ' + x for x in top_3_sim_words]
            print('Sim suggestion list:')
            print(sim_suggestion_list)

            if input_word_count == 1:
                sim_pred = [generate_seq(inspire_model, tokenizer, 27 - 1, x, 3)[0] for x in sim_suggestion_list]
            else:
                sim_pred = [generate_seq(inspire_model, tokenizer, 27 - 1, x, 2)[0] for x in sim_suggestion_list]
            print(f'main pred: {main_pred}, sim pred: {sim_pred}')
            # Make it HTTP friendly
            res = jsonify(main_pred=main_pred, sim_pred=sim_pred)
            return res


handle = open('/Users/jdo/dev/data/garms_nlp/lstm_model/v_ubuntu_3/tokenizer_10.pickle', 'rb')
tokenizer = pickle.load(handle)

inspire_model = load_model('/Users/jdo/dev/data/garms_nlp/lstm_model/v_ubuntu_3/garms_nlp_model_10.h5')
inspire_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print('Model loaded')

# global graph
graph = tf.get_default_graph()

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=80)
