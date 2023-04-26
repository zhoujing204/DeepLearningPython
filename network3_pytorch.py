import pickle 
import gzip
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F


def linear(z): return z 
def ReLU(z): return F.relu(z) 
def sigmoid(z): return F.sigmoid(z) 
def tanh(z): return F.tanh(z)

GPU = True 
if GPU: 
    print("Trying to run under a GPU. If this is not desired, then modify network3.py to set the GPU flag to False.") 
    device = torch.device('cuda') 
else: 
    print("Running with a CPU. If this is not desired, then modify network3.py to set the GPU flag to True.") 
    device = torch.device('cpu')

torch.set_default_tensor_type(torch.FloatTensor) # set default tensor type to float32, equivalent to theano's config.floatX = 'float32'

def load_data_shared(filename="mnist.pkl.gz"):
    """ Load and preprocess the MNIST dataset and returns as a list of shared variables.

    Args: filename (string): The path to the MNIST dataset file which includes training, validation and test data.

    Returns: list: A list of shared variables. Each shared variable is a tuple of input and output.

    Example: 
    >>> data = load_data_shared("mnist.pkl.gz") 
    >>> print(len(data)) # output: 3 
    >>> print(len(data[0])) # output: 2 """
    
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    
    def shared(data):
        """Place the data into shared variables. This allows PyTorch to copy the data to the GPU, if one is available."""
        shared_x = torch.tensor(np.asarray(data[0]),dtype=torch.float32).share_memory_()
        shared_y = torch.tensor(np.asarray(data[1]),dtype=torch.int32).share_memory_()
        return shared_x, shared_y

    return [shared(training_data), shared(validation_data), shared(test_data)]

class Network(object):

    def __init__(self, layers, mini_batch_size):
        """Initialize the neural network with architecture `layers` and takes a value for `mini_batch_size` to be used during training by stochastic gradient descent.
        
        Args:
        layers (List): A list of layers that define the network architecture.
        mini_batch_size (int): The number of examples in each mini-batch.
        
        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = torch.tensor(np.float32(self.mini_batch_size))
        self.y = torch.tensor(np.int32(self.mini_batch_size))
        init_layer = self.layers[0]
        init_layer.forward(self.x, self.x, self.mini_batch_size)
        for j in range(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.forward(prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def SGD(self, training_data, epochs, mini_batch_size, eta, validation_data, test_data, lmbda=0.0):
        """Train the network using mini-batch stochastic gradient descent.
        
        Args:
        training_data (List): A list of training examples and corresponding labels.
        epochs (int): The number of times to loop over the entire training set.
        mini_batch_size (int): The number of examples in each mini-batch.
        eta (float): The learning rate hyperparameter.
        validation_data (List): A list of validation examples and corresponding labels.
        test_data (List): A list of test examples and corresponding labels.
        lmbda (float): The hyperparameter for L2 regularization.
        
        """
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        # compute number of minibatches for training, validation and testing
        num_training_batches = int(len(training_data)/mini_batch_size)
        num_validation_batches = int(len(validation_data)/mini_batch_size)
        num_test_batches = int(len(test_data)/mini_batch_size)

        # define the (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self)+0.5*lmbda*l2_norm_squared/num_training_batches
        grads = torch.autograd.grad(cost, self.params, create_graph=True)
        updates = [(param, param-eta*grad) for param, grad in zip(self.params, grads)]
        
        # define functions for training, validation, and testing
        i = 0
        training_accuracy = []
        validation_accuracy = []
        test_accuracy = []
        while i < epochs:
            i += 1
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                x_mini_batch, y_mini_batch = mini_batch
                x_mini_batch = torch.tensor(np.float32(x_mini_batch))
                y_mini_batch = torch.tensor(np.int64(y_mini_batch))
                cost_ = cost(x_mini_batch, y_mini_batch)
                grads_ = torch.autograd.grad(cost_, self.params)
                updates_ = [(param, param-eta*grad) for param, grad in zip(self.params, grads_)]
                for update in updates_:
                    param, new_param = update[0], update[1]
                    param.data.copy_(new_param)
            training_accuracy.append(self.accuracy(training_data))
            validation_accuracy.append(self.accuracy(validation_data))
            test_accuracy.append(self.accuracy(test_data))
        return training_accuracy, validation_accuracy, test_accuracy

    def accuracy(self, data_set):
        """Returns the accuracy of the model on a given `data_set` of examples."""
        results = [(self.predict(x) == y) for (x, y) in data_set]
        return sum(result for result in results)/len(results)

    def predict(self, x):
        """Predict the output of the network for input `x`."""
        output = self.output
        f = torch.function([self.x], output)
        return np.argmax(f(x))
    
    
class ConvPoolLayer(object): 
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.

    """
    def __init__(self, filter_shape, image_shape, poolsize=(2, 2), activation_fn=F.sigmoid):
        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.

        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.

        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.

        """
        
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn = activation_fn
        self.w = nn.Parameter(torch.randn(filter_shape), requires_grad=True)
        self.b = nn.Parameter(torch.randn(filter_shape[0]), requires_grad=True)

    def forward(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.view(self.image_shape)
        conv_out = F.conv2d(input=self.inpt, weight=self.w, bias=self.b, stride=(1, 1), padding=0)
        pooled_out = F.max_pool2d(conv_out, kernel_size=self.poolsize, stride=self.poolsize)
        self.output = self.activation_fn(pooled_out)
        self.output_dropout = self.output # no dropout in the convolutional layers
        

class FullyConnectedLayer(object): 
    def __init__(self, n_in, n_out, activation_fn=F.sigmoid, p_dropout=0.0): 
        
        self.n_in = n_in 
        self.n_out = n_out 
        self.activation_fn = activation_fn 
        self.p_dropout = p_dropout 
        self.w = nn.Parameter(torch.randn(n_in, n_out) * np.sqrt(1.0 / n_out)) # Initialized weights 
        self.b = nn.Parameter(torch.zeros(n_out)) # Initialized biases
        self.params = [self.w, self.b]

    def forward(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.view(mini_batch_size, self.n_in)
        self.output = self.activation_fn((1-self.p_dropout) * torch.matmul(self.inpt, self.w) + self.b) # dot product of input and weights and apply activation function
        self.y_out = torch.argmax(self.output, dim=1) # index of the highest probability class
        self.inpt_dropout = F.dropout(inpt_dropout.view(mini_batch_size, self.n_in), p=self.p_dropout, training=self.training) # apply dropout to the inputs
        self.output_dropout = self.activation_fn(torch.matmul(self.inpt_dropout, self.w) + self.b) # dot product of input with weights with dropout applied to the inputs
        return self.output, self.output_dropout

    def accuracy(self, y):
        return torch.mean(torch.eq(y, self.y_out).float()) # calculated accuracy


class SoftmaxLayer(object): 
    
    def __init__(self, n_in, n_out, p_dropout=0.0): 
        
        self.n_in = n_in 
        self.n_out = n_out 
        self.p_dropout = p_dropout 
        self.w = nn.Parameter(torch.zeros(n_in, n_out)) # Initialized weights 
        self.b = nn.Parameter(torch.zeros(n_out)) # Initialized biases
        self.params = [self.w, self.b]

    def forward(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.view(mini_batch_size, self.n_in)
        self.output = F.softmax((1-self.p_dropout) * torch.matmul(self.inpt, self.w) + self.b, dim=1) # apply softmax to the dot product of input and weights
        self.y_out = torch.argmax(self.output, dim=1) # index of highest probability class
        self.inpt_dropout = F.dropout(inpt_dropout.view(mini_batch_size, self.n_in), p=self.p_dropout, training=self.training) # apply dropout to the inputs
        self.output_dropout = F.softmax(torch.matmul(self.inpt_dropout, self.w) + self.b, dim=1) # apply softmax to the dot product of the inputs with dropout applied
        return self.output, self.output_dropout

    def cost(self, net):
        return -torch.mean(torch.log(self.output_dropout)[torch.arange(net.y.shape[0]), net.y]) # calculate log-likelihood cost

    def accuracy(self, y):
        return torch.mean(torch.eq(y, self.y_out).float()) # calculate accuracy


#### Miscellanea

def size(data):
    "Return the size of the dataset `data`."
    return data[0].shape[0]

def dropout_layer(layer, p_dropout):
    mask = torch.bernoulli(torch.ones_like(layer) - p_dropout)  # generate a binary dropout mask for given layer
    return layer * mask  # apply the dropout mask to the layer inputs










