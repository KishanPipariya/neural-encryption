import tensorflow as tf
import numpy as np
import random
import keras
from sklearn.preprocessing import OneHotEncoder


tf.random.set_seed(42)
random.seed(0)
np.random.seed(0)


b = bin(10)
print(b, type(b))

def alph_to_bin(al):
    ch = bin(al)[2:]
    ch = list(ch)
    ch = list(map(int, ch))
    return ch

print(alph_to_bin(ord('a')))

for i in range(ord('a'), ord('z') + 1):
    print(chr(i), alph_to_bin(i))

input_shape = (7,)

model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(16, input_shape=input_shape, activation='relu' ))
model.add(tf.keras.layers.Dense(7, activation='linear'))
model.summary()

input_tensor = tf.convert_to_tensor(alph_to_bin(ord('a')), dtype=tf.float32)
input_tensor = tf.reshape(input_tensor, (7,))
print(model.predict(tf.expand_dims(input_tensor, axis=0)))

model.save('enc_model.keras')

new_model = tf.keras.models.load_model('enc_model.keras')

print(new_model.predict(tf.expand_dims(input_tensor, axis=0)))

