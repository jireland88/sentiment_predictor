from sentiment_predictor import *
import matplotlib.pyplot as plt

sentences = ["I love playing squash", "I hate playing tennis", "You kind of like basketball"]
outputs = [10, -10, 6]

sp = SentimentPredictor(sentences, outputs)

word_weights = sp.get_word_weights()

x = []
y = []

for i in word_weights:
    x.append(i[0])
    y.append(i[1])

plt.bar(x, y)
plt.ylabel("Sentiment")
plt.xlabel("Word")
plt.show()
