import tensorflow as tf
from tensorflow import keras

sentences = [
    'I love my dog',
    'I love my cat', 
    'hello world', 
    'how are you doing',
    'I am fine thank you',
    'machine learnign is fun'
]

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = 100, oov_token="<OOV>")

tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post', 
                                                        truncating='post')

print(word_index)
print(sequences)
print(padded)