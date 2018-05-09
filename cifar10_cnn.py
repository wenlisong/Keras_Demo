import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation
from keras.datasets import cifar10
from keras.utils import to_categorical

batch_size = 32
num_classes = 10
epochs = 10

# load cifar10 dateset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print(x_train.shape[0], 'train examples')
print(x_test.shape[0], 'test examples')

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Model
model = Sequential()

# add some layers
model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

opt = keras.optimizers.sgd(lr=0.01, momentum=0.9,decay=0.5)

model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

# configure learning process
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
