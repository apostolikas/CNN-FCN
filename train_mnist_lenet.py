import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from lenet5 import LeNet
from read_dataset import read_mnist
from gpu import set_gpu_config
from keras import backend as K


set_gpu_config()
print(tf.test.is_gpu_available())

np.random.seed(0)
tf.random.set_seed(0)


if __name__ == "__main__":

    epochs = 10
    batch_size = 128

    train = {}
    test = {}
    train['images'], train['labels'] = read_mnist('./data/train-images-idx3-ubyte.gz', './data/train-labels-idx1-ubyte.gz')
    test['images'], test['labels'] = read_mnist('./data/t10k-images-idx3-ubyte.gz', './data/t10k-labels-idx1-ubyte.gz')
    print('# of training images:', train['images'].shape[0])
    print('# of test images:', test['images'].shape[0])

    # zero pad
    train['images'] = np.pad(train['images'], ((0,0),(2,2),(2,2),(0,0)), 'constant')
    test['images'] = np.pad(test['images'], ((0,0),(2,2),(2,2),(0,0)), 'constant')

    # one hot
    x_train , y_train = train['images']/255, to_categorical(train['labels'])
    x_test , y_test = test['images']/255, to_categorical(test['labels'])


    model = LeNet()
    history = model.train(x_train, y_train, batch_size, epochs)
    print("Training completed")
    model.evaluate(x_test, y_test, batch_size)

    train_output = model.get_output([x_train])[0]
    test_output = model.get_output([x_test])[0]
    np.savetxt("train-adam.csv", train_output, delimiter=",")
    np.savetxt("test-adam.csv", test_output, delimiter=",")
