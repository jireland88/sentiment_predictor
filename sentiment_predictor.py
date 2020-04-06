import pandas as pd
from sklearn.linear_model import LinearRegression

# This module takes in an array of setences (minus all punctuation etc), and a real output,
# then returns an SKLearn linear regression.


class SentimentPredictor(object):
    def __init__(self, sentences, outputs):
        self.sentences = sentences
        self.outputs = outputs
        self.words = self.build_words_list()
        self.input_df = self.build_input_df()
        self.model = self.produce_model()


    # creates a list containing all words
    def build_words_list(self):
        words = []

        for sentence in self.sentences:
            words_in_sentence = sentence.split()
            for word in words_in_sentence:
                if word not in words:
                    words.append(word)

        return words

    # creates a row vector containing the number of times each word in the word list occurs in the sentence
    def get_row_vector_for_sentence(self, sentence):
        words_in_sentence = sentence.split()

        row = { i : 0 for i in self.words }

        for word in words_in_sentence:
            row[word] += 1

        return row

    def build_input_df(self):
        input_df = pd.DataFrame(columns=self.words)

        for sentence in self.sentences:
            row = self.get_row_vector_for_sentence(sentence)
            input_df = input_df.append(row, ignore_index=True)

        return input_df

    def produce_model(self):
        clf = LinearRegression().fit(self.input_df, self.outputs)
        return clf

    def get_sentiment(self, sentence):
        input = self.get_row_vector_for_sentence(sentence)
        return self.model.predict(input)

    def get_word_weights(self):
        coef = self.model.coef_
        word_weights = []

        for i in range(0, len(self.words)):
            word_weights.append([self.words[i], coef[i]])

        return word_weights
