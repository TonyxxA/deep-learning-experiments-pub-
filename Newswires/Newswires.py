from tensorflow.keras.datasets import reuters

sample_newswire = reuters.load_data()[0][0][2]

word_index = reuters.get_word_index()
reversed_index = dict([(value, key) for (key, value) in word_index.items()])


decoded_newswire = " ".join([reversed_index.get(i - 3, "|") for i in sample_newswire[:-2]])

print(decoded_newswire)
