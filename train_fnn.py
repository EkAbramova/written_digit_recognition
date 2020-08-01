import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable

from models import fnn_model
from models import cnn_pytorch


## load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#plt.figure()
#plt.imshow(x_train[13], cmap='Greys')

# TRAIN FNN

# layers parameters
input_size = 28 * 28 * 1
hidden_size = 100
output_size = 10

# normalize data
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(x_train.shape[0], input_size)
x_test = x_test.reshape(x_test.shape[0], input_size)

# train FNN model
model = fnn_model.TwoLayerNet(weights=0, mode='learn', input_size=input_size,
                              hidden_size=hidden_size,
                              output_size=10)

stats = model.train(x_train, y_train, x_test, y_test, num_iters=10000)
val_acc = (model.predict(x_test) == y_test).mean()

#print('Validation accuracy: ', val_acc)
##Validation accuracy:  0.8505

plt.plot(stats['loss_history'])
plt.title('Loss history')
plt.legend()


plt.plot(stats['train_acc_history'], label='train accuracy')
plt.plot(stats['val_acc_history'], label='validation accuracy')
plt.title('FNN accuracy')
plt.legend()

# save parameters
np.save('models/updated_weights.npy', model.params)
np.save('models/original_weights.npy', model.params)

