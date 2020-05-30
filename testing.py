from models import fnn_model
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
#from sklearn.model_selection import train_test_split

#sns.set()

import tensorflow as tf

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

x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(x_train.shape[0], input_size)
x_test = x_test.reshape(x_test.shape[0], input_size)

model = fnn_model.TwoLayerNet(weights=0, mode='learn', input_size=input_size,
                              hidden_size=hidden_size,
                              output_size=10)

stats = model.train(x_train, y_train, x_test, y_test, num_iters=10000)

val_acc = (model.predict(x_test) == y_test).mean()
print('Validation accuracy: ', val_acc)
##Validation accuracy:  0.8505

plt.plot(stats['loss_history'])
plt.title('Loss history')
plt.legend()


plt.plot(stats['train_acc_history'], label='train accuracy')
plt.plot(stats['val_acc_history'], label='validation accuracy')
plt.title('FNN accuracy')
plt.legend()


np.save('models/updated_weights.npy', model.params)
np.save('models/original_weights.npy', model.params)

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

print(model.predict(x_test[14:15]))

model.predict(x_test[20:21])



################


def fit(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters())#,lr=0.001, betas=(0.9,0.999))
    error = nn.CrossEntropyLoss()
    EPOCHS = 5
    model.train()
    for epoch in range(EPOCHS):
        correct = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            var_X_batch = Variable(X_batch).float()
            var_y_batch = Variable(y_batch)
            optimizer.zero_grad()
            output = model(var_X_batch)
            loss = error(output, var_y_batch)
            loss.backward()
            optimizer.step()

            # Total correct predictions
            predicted = torch.max(output.data, 1)[1]
            correct += (predicted == var_y_batch).sum()
            #print(correct)
            if batch_idx % 50 == 0:
                print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                    epoch, batch_idx*len(X_batch), len(train_loader.dataset), 100.*batch_idx / len(train_loader), loss.item(), float(correct*100) / float(BATCH_SIZE*(batch_idx+1))))


def evaluate(model):
    correct = 0
    for test_imgs, test_labels in test_loader:
        #print(test_imgs.shape)
        test_imgs = Variable(test_imgs).float()
        output = model(test_imgs)
        predicted = torch.max(output,1)[1]
        correct += (predicted == test_labels).sum()
    print("Test accuracy:{:.3f}% ".format( float(correct) / (len(test_loader)*BATCH_SIZE)))

cnn = CNN()

df = pd.read_csv('models/train.csv')
BATCH_SIZE = 32

y = df['label'].values
X = df.drop(['label'],1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

torch_X_train = torch.from_numpy(X_train).type(torch.LongTensor)
torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor) # data type is long

# create feature and targets tensor for test set.
torch_X_test = torch.from_numpy(X_test).type(torch.LongTensor)
torch_y_test = torch.from_numpy(y_test).type(torch.LongTensor)
##torch_y_test = torch.from_numpy(y_test).type(torch.LongTensor) # data type is long

##v-
torch_X_train = torch_X_train.view(-1, 1, 28, 28).float()/255.0
torch_X_test = torch_X_test.view(-1, 1, 28, 28).float()/255.0

print(torch_X_train.shape)
print(torch_X_test.shape)

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(torch_X_train,torch_y_train)
test = torch.utils.data.TensorDataset(torch_X_test,torch_y_test)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = BATCH_SIZE, shuffle = False)

fit(cnn, train_loader)

evaluate(cnn)


torch.save(cnn.state_dict(), 'models/cnn_mnist.pt')

#################
## confusion matrix
