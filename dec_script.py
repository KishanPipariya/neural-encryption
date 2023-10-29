import tensorflow as tf
import numpy as np
import random
import keras
import sys

tf.random.set_seed(42)
random.seed(0)
np.random.seed(0)

model_dec = tf.keras.models.load_model("dec_model.keras")

params = sys.argv[1].split(" ")
l = len(params)
print(params)
params = list(map(float, params))
for i in range(0, l, 7):
    input_tensor = tf.convert_to_tensor(params[i:i+7], dtype=tf.float32)
    input_tensor = tf.reshape(input_tensor, (7,))
    prediction = model_dec.predict(tf.expand_dims(input_tensor, axis=0))[0]
    m = np.argmax(prediction)
    print(chr(int(m)-1))





# def alph_to_bin(al):
#     ch = bin(al)[2:]
#     ch = list(ch)
#     ch = list(map(int, ch))
#     return ch

# model = tf.keras.models.load_model('enc_model.keras')

# results = tf.zeros((1, 7))

# for i in range(ord('a'), ord('z') + 1):
#     input_tensor = tf.convert_to_tensor(alph_to_bin(i), dtype=tf.float32)
#     input_tensor = tf.reshape(input_tensor, (7,))
#     ans1 = model.predict(tf.expand_dims(input_tensor, axis=0))
#     results = tf.concat([results, ans1], axis=0)

# results = results[1:]


# model_dec = keras.Sequential([
#     keras.layers.Dense(128, activation='relu', input_shape=(7,)),
#     keras.layers.Dense(64, activation='relu'),
#     keras.layers.Dense(26, activation='softmax')
# ])
# model_dec.summary()

# model_dec.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# labels = []
# for i in range(26):
#     l = [0] * 26
#     l[i] = 1
#     labels.append(l)
# labels = tf.convert_to_tensor(labels)
# labels.shape



# epochs = 1000
# model_dec.fit(results, labels, epochs=epochs)

# model_dec.save('dec_model.keras')