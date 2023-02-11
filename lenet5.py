import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K


class LeNet:
    def __init__(self, output_sizes=10):
        super().__init__()
        self.model = Sequential()
        self.model.add(Conv2D(filters=6, kernel_size=(5, 5), activation='tanh', input_shape=(32,32,1)))
        self.model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='tanh'))
        self.model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(units=120, activation='tanh'))
        self.model.add(Dense(units=84, activation='tanh'))
        self.model.add(Dense(units=output_sizes, activation = 'softmax'))
        self.model.summary()
        self.model.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer=keras.optimizers.Adam(), metrics=['accuracy'])


    def train(self, x_train, y_train, batch_size, epochs):
        
        history = self.model.fit(x_train, y_train, batch_size, epochs, verbose=1)
        return history

    def evaluate(self, x, y, batch_size):

        results = self.model.evaluate(x, y, batch_size)
        print("The test accuracy of the model is :" ,results[1])
        return results
    
    def get_output(self, x):
        get_6th_layer_output = K.function([self.model.layers[0].input],[self.model.layers[6].output])
        x_output = get_6th_layer_output(x)

        return x_output

