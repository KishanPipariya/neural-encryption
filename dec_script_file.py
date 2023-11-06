import tensorflow as tf
import numpy as np
import random
import keras
import sys

#Setting up reproductibility for random number generation
tf.random.set_seed(42)
random.seed(0)
np.random.seed(0)

#Loading the pre-trained model
model_dec = tf.keras.models.load_model("dec_model.keras")

#Getting the path from commandline argument
params = sys.argv[1]
f = open(params)
content = f.read()
content = content.split(' ')
content = content[:-1]
l = len(content)
print(params)
content = list(map(float, content))
f = open('dec.txt', 'w')
for i in range(0, l, 7):
    input_tensor = tf.convert_to_tensor(content[i:i+7], dtype=tf.float32)
    input_tensor = tf.reshape(input_tensor, (7,))
    prediction = model_dec.predict(tf.expand_dims(input_tensor, axis=0))[0]
    m = np.argmax(prediction)
    ch = chr(int(m)-1)
    f.write(ch)