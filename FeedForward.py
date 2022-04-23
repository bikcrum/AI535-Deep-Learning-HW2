from turtle import width
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib

font = {'weight': 'normal', 'size': 22}
matplotlib.rc('font', **font)
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


######################################################
# Q4 Implement Init, Forward, and Backward For Layers
######################################################


class CrossEntropySoftmax:

    # Compute the cross entropy loss after performing softmax
    # logits -- batch_size x num_classes set of scores, logits[i,j] is score of class j for batch element i
    # labels -- batch_size x 1 vector of integer label id (0,1,2) where labels[i] is the label for batch element i
    #
    # Output should be a positive scalar value equal to the average cross entropy loss after softmax
    def forward(self, logits, labels):
        # raise Exception('Student error: You haven\'t implemented the forward pass for CrossEntropySoftmax yet.')
        self.logits = logits
        self.labels = labels
        # select corresponding probabilities indexed by labels for each example and sum it
        self.loss = -np.mean(np.log(logits[np.arange(len(logits)), labels]))
        return self.loss

    # Compute the gradient of the cross entropy loss with respect to the the input logits
    def backward(self):
        # raise Exception('Student error: You haven\'t implemented the backward pass for CrossEntropySoftmax yet.')
        self.logits[np.arange(len(self.logits)), self.labels] -= 1
        return self.logits / len(self.logits)


class ReLU:

    # Compute ReLU(input) element-wise
    def forward(self, input, W=None, b=None):
        # raise Exception('Student error: You haven\'t implemented the forward pass for ReLU yet.')
        self.act = np.maximum(0, input)
        return self.act

    # Given dL/doutput, return dL/dinput
    def backward(self, grad):
        # raise Exception('Student error: You haven\'t implemented the backward pass for ReLU yet.')
        doutput_dinput = self.act > 0

        return grad * doutput_dinput

    # No parameters so nothing to do during a gradient descent step
    def step(self, step_size):
        return


class AdamOptimizer:
    def __init__(self, beta=0.9, weight_decay=0, epsilon=1e-8):
        self.beta = beta
        self.weight_decay = weight_decay
        self.epsilon = epsilon
        self.m, self.v = 0, 0

    def get_step(self, lr, f, df):
        self.m = self.beta * self.m + (1 - self.beta) * (df + self.weight_decay * f)
        self.v = self.beta * self.v + (1 - self.beta) * (df + self.weight_decay * f) ** 2

        _m = self.m / (1 - self.beta)
        _v = self.v / (1 - self.beta)

        return lr * _m / (np.sqrt(_v) + self.epsilon)


class LinearLayer:
    # Initialize our layer with (input_dim, output_dim) weight matrix and a (1,output_dim) bias vector
    def __init__(self, input_dim, output_dim):
        # raise Exception('Student error: You haven\'t implemented the init for LinearLayer yet.')
        self.weight_optimizer = AdamOptimizer()
        self.bias_optimizer = AdamOptimizer()

        self.W = np.random.random((input_dim, output_dim)) * np.sqrt(2 / input_dim)
        self.b = np.zeros(shape=(1, output_dim))

    # During the forward pass, we simply compute XW+b
    def forward(self, input):
        # raise Exception('Student error: You haven\'t implemented the forward pass for LinearLayer yet.')
        self.input = input
        self.act = input @ self.W + self.b
        return self.act

    # Inputs:
    #
    # grad dL/dZ -- For a batch size of n, grad is a (n x output_dim) matrix where
    #         the i'th row is the gradient of the loss of example i with respect
    #         to z_i (the output of this layer for example i)

    # Computes and stores:
    #
    # self.grad_weights dL/dW --  A (input_dim x output_dim) matrix storing the gradient
    #                       of the loss with respect to the weights of this layer.
    #                       This is an summation over the gradient of the loss of
    #                       each example with respect to the weights.
    #
    # self.grad_bias dL/db --     A (1 x output_dim) matrix storing the gradient
    #                       of the loss with respect to the bias of this layer.
    #                       This is an summation over the gradient of the loss of
    #                       each example with respect to the bias.

    # Return Value:
    #
    # grad_input dL/dX -- For a batch size of n, grad_input is a (n x input_dim) matrix where
    #               the i'th row is the gradient of the loss of example i with respect
    #               to x_i (the input of this layer for example i)

    def backward(self, grad):
        # raise Exception('Student error: You haven\'t implemented the backward pass for LinearLayer yet.')
        self.grad_weights = self.input.T @ grad
        self.grad_bias = grad.sum(axis=0).reshape(1, -1)
        grad_input = grad @ self.W.T
        return grad_input

    ######################################################
    # Q5 Implement ADAM with Weight Decay
    ######################################################
    def step(self, step_size):
        # raise Exception('Student error: You haven\'t implemented the step for LinearLayer yet.')
        self.W = self.W - step_size * self.weight_optimizer.get_step(step_size, self.W, self.grad_weights)
        self.b = self.b - step_size * self.bias_optimizer.get_step(step_size, self.b, self.grad_bias)


######################################################
# Q6 Implement Evaluation and Training Loop
######################################################

# Given a model, X/Y dataset, and batch size, return the average cross-entropy loss and accuracy over the set
def evaluate(model, X_val, Y_val, batch_size):
    # raise Exception('Student error: You haven\'t implemented the step for evalute function.')
    Y_pred = model.forward(X_val)

    logits = softmax(Y_pred)
    loss = CrossEntropySoftmax().forward(logits, Y_val.flatten())

    true_count = np.count_nonzero([Y_pred.argmax(axis=1) == Y_val.flatten()])

    return loss, true_count / len(X_val)


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1).reshape(-1, 1))
    return e_x / e_x.sum(axis=1).reshape(-1, 1)


def main():
    # Set optimization parameters (NEED TO CHANGE THESE)
    batch_size = 32
    max_epochs = 100
    step_size = 0.001

    number_of_layers = 2
    width_of_layers = 32

    load_trained_model = True

    # Load data
    X_train, Y_train, X_val, Y_val, X_test, Y_test = loadCIFAR10Data()

    # Some helpful dimensions
    num_examples, input_dim = X_train.shape
    output_dim = 3  # number of class labels

    # Build a network with input feature dimensions, output feature dimension,
    # hidden dimension, and number of layers as specified below. You can edit this as you please.
    cross_entropy = CrossEntropySoftmax()
    net = FeedForwardNeuralNetwork(input_dim, output_dim, width_of_layers, number_of_layers)

    if load_trained_model:
        with open(f'model_params-{number_of_layers}-{width_of_layers}.pkl', 'rb') as inp:
            weights, biases = pickle.load(inp)
            net.apply_params(weights, biases)

    # Some lists for book-keeping for plotting later
    losses = []
    val_losses = []
    accs = []
    val_accs = []

    # raise Exception('Student error: You haven\'t implemented the training loop yet.')

    # For each epoch below max epochs
    for epoch in range(max_epochs):
        # Scramble order of examples
        # Use these to fetch both x and y train in random order
        indices = list(range(len(X_train)))
        np.random.shuffle(indices)

        # for each batch in data:
        for i in range(0, len(X_train), batch_size):
            # Gather batch
            inputs = X_train[indices][i:i + batch_size] / 255.0
            labels = Y_train[indices][i:i + batch_size]

            # Predict
            outputs = net.forward(inputs)

            # Compute loss
            logits = softmax(outputs)
            loss = cross_entropy.forward(logits, labels.flatten())

            # Backward loss and networks
            loss_grad = cross_entropy.backward()
            net.backward(loss_grad)

            # Take optimizer step
            net.step(step_size)

            # Book-keeping for loss / accuracy
            losses.append(loss)

            true_count = np.count_nonzero([outputs.argmax(axis=1) == labels.flatten()])
            accs.append(true_count / batch_size)

        # Evaluate performance on validation.

        ###############################################################
        # Print some stats about the optimization process after each epoch
        ###############################################################
        # epoch_avg_loss -- average training loss across batches this epoch
        # epoch_avg_acc -- average accuracy across batches this epoch
        epoch_avg_loss, epoch_avg_acc = evaluate(net, X_train / 255.0, Y_train, batch_size)
        # vacc -- validation accuracy this epoch
        vloss, vacc = evaluate(net, X_val / 255.0, Y_val, batch_size)
        val_losses.append(vloss)
        val_accs.append(vacc)
        ###############################################################

        logging.info(
            "[Epoch {:3}]   Loss:  {:8.4}     Train Acc:  {:8.4}%      Val Acc:  {:8.4}%".format(epoch, epoch_avg_loss,
                                                                                                 epoch_avg_acc,
                                                                                                 vacc))

    ###############################################################
    # Code for producing output plot requires
    ###############################################################
    # losses -- a list of average loss per batch in training
    # accs -- a list of accuracies per batch in training
    # val_losses -- a list of average validation loss at each epoch
    # val_acc -- a list of validation accuracy at each epoch
    # batch_size -- the batch size
    ################################################################

    # Plot training and validation curves
    fig, ax1 = plt.subplots(figsize=(16, 9))
    color = 'tab:red'
    ax1.plot(range(len(losses)), losses, c=color, alpha=0.25, label="Train Loss")
    ax1.plot([np.ceil((i + 1) * len(X_train) / batch_size) for i in range(len(val_losses))], val_losses, c="red",
             label="Val. Loss")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Avg. Cross-Entropy Loss", c=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(-0.01, 3)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.plot(range(len(losses)), accs, c=color, label="Train Acc.", alpha=0.25)
    ax2.plot([np.ceil((i + 1) * len(X_train) / batch_size) for i in range(len(val_accs))], val_accs, c="blue",
             label="Val. Acc.")
    ax2.set_ylabel(" Accuracy", c=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(-0.01, 1.01)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.legend(loc="center")
    ax2.legend(loc="center right")
    plt.show()

    ################################
    # Q7 Tune and Evaluate on Test
    ################################
    _, tacc = evaluate(net, X_test / 255.0, Y_test, batch_size)
    print(tacc)

    # Save model params to resume training
    with open(f'model_params-{number_of_layers}-{width_of_layers}.pkl', 'wb') as out:
        pickle.dump(net.get_params(), out)


#####################################################
# Feedforward Neural Network Structure
# -- Feel free to edit when tuning
#####################################################

class FeedForwardNeuralNetwork:

    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, _net=None):

        if num_layers == 1:
            self.layers = [LinearLayer(input_dim, output_dim)]
        else:
            self.layers = [LinearLayer(input_dim, hidden_dim)]
            self.layers.append(ReLU())
            for i in range(num_layers - 2):
                self.layers.append(LinearLayer(hidden_dim, hidden_dim))
                self.layers.append(ReLU())
            self.layers.append(LinearLayer(hidden_dim, output_dim))

    def get_params(self):
        return list(zip(*[(layer.W, layer.b) for layer in self.layers if type(layer) is LinearLayer]))

    def apply_params(self, weights, biases):
        for i, layer in enumerate(filter(lambda l: type(l) is LinearLayer, self.layers)):
            layer.W = weights[i]
            layer.b = biases[i]

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def step(self, step_size):
        for layer in self.layers:
            layer.step(step_size)


#####################################################
# Utility Functions for Loading and Visualizing Data
#####################################################

def loadCIFAR10Data():
    with open("cifar10_hst_train", 'rb') as fo:
        data = pickle.load(fo)
    X_train = data['images']
    Y_train = data['labels']

    with open("cifar10_hst_val", 'rb') as fo:
        data = pickle.load(fo)
    X_val = data['images']
    Y_val = data['labels']

    with open("cifar10_hst_test", 'rb') as fo:
        data = pickle.load(fo)
    X_test = data['images']
    Y_test = data['labels']

    logging.info("Loaded train: " + str(X_train.shape))
    logging.info("Loaded val: " + str(X_val.shape))
    logging.info("Loaded test: " + str(X_test.shape))

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def displayExample(x):
    r = x[:1024].reshape(32, 32)
    g = x[1024:2048].reshape(32, 32)
    b = x[2048:].reshape(32, 32)

    plt.imshow(np.stack([r, g, b], axis=2))
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
