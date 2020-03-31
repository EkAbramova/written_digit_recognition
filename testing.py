from models import fnn_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

import tensorflow as tf


## Xavier initialization
## prediction for single
## pic for confusion matrix ??

#new module
# create model, train, save weights



(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


plt.figure()
plt.imshow(x_train[13], cmap='Greys')

input_size = 28 * 28 * 1
hidden_size = 100
output_size = 10

x_train = x_train.reshape(x_train.shape[0], input_size)
x_test = x_test.reshape(x_test.shape[0], input_size)

model = fnn_model.TwoLayerNet(input_size=input_size,
                              hidden_size=hidden_size,
                              output_size=10)

stats = model.train(x_train, y_train, x_test, y_test, num_iters=10000)





plt.figure(figsize=(20, 5))
#plt.subplot(2, 1, 1)
plt.plot(stats['loss_history'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

#plt.subplot(2, 1, 2)
plt.figure(figsize=(20, 5))
plt.plot(stats['train_acc_history'], label='train')
plt.plot(stats['val_acc_history'], label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Clasification accuracy')
plt.legend()
plt.show()



plt.figure()
plt.imshow(x_test[14].reshape(28, 28), cmap='Greys')

print(model.predict(x_test[13:15]))






