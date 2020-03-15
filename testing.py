from models import fnn_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train = pd.read_csv("train.csv")

Y_train = train["label"]
# Drop 'label' column
X_train = train.drop(labels = ["label"],axis = 1)
# free some space
del train

# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
X_train_r = X_train.values.reshape(42000, -1)


input_size = 32 * 32 * 3
hidden_size = 100
output_size = 10


model = fnn_model.TwoLayerNet(input_size=input_size,
                              hidden_size=hidden_size,
                              output_size=10)

model.train(X_train_r, Y_train, X_train_r, Y_train)

preds = model.predict(X_train_r)

train_acc = (model.predict(X_train_r) == Y_train).mean()
print(preds)
print(Y_train)

pixels = X_train_r[10]

# Make those columns into a array of 8-bits pixels
# This array will be of 1D with length 784
# The pixel intensity values are integers from 0 to 255
pixels = np.array(pixels, dtype='uint8')
pixels = pixels.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()






