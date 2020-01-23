import tensorflow as tf
import fnn_model
import numpy as np


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train[:1000]
y_train = y_train[:1000]
x_test = x_test[:1000]
y_test = y_test[:1000]

input_size = 28 * 28 #* 3
hidden_size = 100
output_size = 10


weights = {}
weights['W1'] = 0.00001 * np.random.randn(input_size, hidden_size)
weights['b1'] = np.zeros(hidden_size)
weights['W2'] = 0.00001 * np.random.randn(hidden_size, output_size)
weights['b2'] = np.zeros(output_size)


x_train_r = x_train.reshape(1000, -1)
x_train_r.shape

x_test_r = x_test.reshape(1000, -1)

model = fnn_model.FNN(weights)

model.train(x_train_r, y_train)

preds = model.predict(x_test_r)

train_acc = (model.predict(x_test_r) == y_train).mean()
print(train_acc)





