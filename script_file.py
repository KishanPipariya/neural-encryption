import tensorflow as tf
import numpy as np
import random
import sys

name = sys.argv[0]
l = len(sys.argv)
param = sys.argv[1]

tf.random.set_seed(42)
random.seed(0)
np.random.seed(0)

def alph_to_bin(al):
    ch = bin(al)[2:]
    ch = list(ch)
    ch = list(map(int, ch))
    return ch

new_model = tf.keras.models.load_model('enc_model.keras')

f = open(param)
content = f.read()

f1 = open("enc.txt", "w")
for i in content:
    input_tensor = tf.convert_to_tensor(alph_to_bin(ord(i)), dtype=tf.float32)
    input_tensor = tf.reshape(input_tensor, (7,))
    pred = new_model.predict(tf.expand_dims(input_tensor, axis=0))[0]
    print(type(pred))
    for i in pred:
        f1.write(str(i)+" ")
    # print(new_model.predict(tf.expand_dims(input_tensor, axis=0))[0])




