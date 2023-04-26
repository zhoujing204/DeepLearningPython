# ----------------------
# - read the input data:

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

# ----------------------
# - network3.py example:
import network3_pytorch as nwpytorch
from network3_pytorch import Network, ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer # softmax plus log-likelihood cost is more common in modern image classification networks.

# read data:
training_data, validation_data, test_data = nwpytorch.load_data_shared()
# mini-batch size:
mini_batch_size = 10

# chapter 6 - shallow architecture using just a single hidden layer, containing 100 hidden neurons.

net = Network([
    FullyConnectedLayer(n_in=784, n_out=100),
    SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)


# chapter 6 - 5x5 local receptive fields, 20 feature maps, max-pooling layer 2x2
'''
net = Network([
    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                  filter_shape=(20, 1, 5, 5),
                  poolsize=(2, 2)),
    FullyConnectedLayer(n_in=20*12*12, n_out=100),
    SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)
'''

# chapter 6 - inserting a second convolutional-pooling layer to the previous example => better accuracy
'''
net = Network([
    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                  filter_shape=(20, 1, 5, 5),
                  poolsize=(2, 2)),
    ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                  filter_shape=(40, 20, 5, 5),
                  poolsize=(2, 2)),
    FullyConnectedLayer(n_in=40*4*4, n_out=100),
    SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)
'''

# chapter 6 -  rectified linear units and some l2 regularization (lmbda=0.1) => even better accuracy
# from network3 import ReLU
# net = Network([
#     ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
#                   filter_shape=(20, 1, 5, 5),
#                   poolsize=(2, 2),
#                   activation_fn=ReLU),
#     ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
#                   filter_shape=(40, 20, 5, 5),
#                   poolsize=(2, 2),
#                   activation_fn=ReLU),
#     FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
#     SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
# net.SGD(training_data, 60, mini_batch_size, 0.03, validation_data, test_data, lmbda=0.1)






