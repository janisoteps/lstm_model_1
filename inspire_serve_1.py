import pickle
import jsonlines
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from flask import Flask, jsonify, request
import tensorflow as tf
sess = tf.Session()


# # # # Flask Itself # # # #
application = app = Flask(__name__)

word_counts_set = set()
input_file = jsonlines.open('/Users/jdo/dev/data/garms_nlp/lstm_model/v_ubuntu_2/sanitized_list_4.jsonl', 'r')
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


@app.route("/api/sequences", methods=["POST"])
def features():
    if request.method == "POST":
        data = request.get_json(force=True)
        print('Data: ', str(request.data))
        word_array = data['word_array']
        req_text = data['text']

        prediction = generate_seq(inspire_model, tokenizer, 27 - 1, req_text, 3)

        # Make it HTTP friendly
        res = jsonify(res=result_dict)

        return res


handle = open('/Users/jdo/dev/data/garms_nlp/lstm_model/v_ubuntu_3/tokenizer_10.pickle', 'rb')
tokenizer = pickle.load(handle)

inspire_model = load_model('/Users/jdo/dev/data/garms_nlp/lstm_model/v_ubuntu_3/garms_nlp_model_10.h5')

inspire_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# global graph
graph = tf.get_default_graph()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=80)
