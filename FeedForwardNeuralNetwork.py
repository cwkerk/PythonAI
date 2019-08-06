from matplotlib import pyplot
import numpy
import numpy.linalg as linarg
import numpy.random as random
import time


def sigmoid(z):
    return 1 / (1 + numpy.exp(-z))


def sigmoid_derivative(a):
    return a * (1 - a)


class FullConnectedNetwork:

    """Full connected feed forward neural network"""

    error_tolerance = 1e-10
    max_epoch = 1e10
    momentum = 1

    def __init__(self, sizes, **kwargs):
        # size = number of neurons in ith layer
        self.sizes = sizes
        try:
            self.activation_function = kwargs.pop("activation_function")
            self.activation_function_derivative = kwargs.pop("activation_function_derivative")
        except:
            self.activation_function = sigmoid
            self.activation_function_derivative = sigmoid_derivative
        try:
            self.biases = kwargs.pop("biases")
        except:
            self.biases = numpy.array([random.randn(size) for size in sizes[1:]])
        try:
            self.weights = kwargs.pop("weights")
        except:
            self.weights = numpy.array([random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])])
        self.nabla_biases = numpy.array([numpy.zeros(bias.shape) for bias in self.biases])
        self.nabla_weights = numpy.array([numpy.zeros(weight.shape) for weight in self.weights])

    def _back_propagation(self, error, outputs):
        nabla_biases = numpy.array([numpy.zeros(bias.shape) for bias in self.biases])
        nabla_weights = numpy.array([numpy.zeros(weight.shape) for weight in self.weights])
        for i in range(1, len(self.sizes)):
            # delta = ∂(error)/∂(last_output)
            #       = ∂(error)/∂(last_activation) * ∂(last_activation)/∂(last_output)
            if i == 1:
                delta = error * self.activation_function_derivative(outputs[-1])
            else:
                delta = numpy.dot(self.weights[-i+1].T, delta) * self.activation_function_derivative(outputs[-i])
            # ∂(error)/∂(last_bias) = ∂(error)/∂(last_output)
            nabla_biases[-i] = delta
            # ∂(error)/∂(last_weight) = ∂(error)/∂(last_output) * ∂(last_output)/∂(last_weight)
            nabla_weights[-i] = numpy.outer(delta, outputs[-i-1])
        return nabla_biases, nabla_weights

    def feed_forward(self, inputs):
        a = inputs
        outputs = [inputs]
        for (b, w) in zip(self.biases, self.weights):
            z = numpy.dot(w, a) + b
            a = self.activation_function(z)
            outputs.append(a)
        return numpy.array(outputs)

    def full_batch_gradient_descent_training(self, inputs, labels, *kwargs):
        norm_inputs = inputs / linarg.norm(inputs)
        norm_labels = inputs / linarg.norm(labels)
        try:
            momentum = kwargs.pop("momentum")
        except:
            momentum = self.momentum
        epoch = 0
        epoch_error = self.error_tolerance
        epoch_errors = []
        epoch_size = len(norm_inputs)
        while epoch < self.max_epoch and epoch_error >= self.error_tolerance:
            epoch += 1
            epoch_error = 0
            batch_errors = []
            batch_outputs = []
            for i in range(epoch_size):
                output = self.feed_forward(norm_inputs[i])
                error = output[-1] - norm_labels[i]
                batch_errors.append(error)
                batch_outputs.append(output)
                epoch_error += 0.5 * sum(error * error)
            epoch_error /= epoch_size
            epoch_errors.append(epoch_error)
            for i in range(epoch_size):
                self.nabla_biases, self.nabla_weights = self._back_propagation(batch_errors[i], batch_outputs[i])
                self.biases -= momentum * self.nabla_biases
                self.weights -= momentum * self.nabla_weights
        return numpy.array(epoch_errors)

    def mini_batch_gradient_descent_training(self):
        pass

    def stochastic_gradient_descent_training(self, inputs, labels, *kwargs):
        norm_inputs = inputs / linarg.norm(inputs)
        norm_labels = inputs / linarg.norm(labels)
        try:
            momentum = kwargs.pop("momentum")
        except:
            momentum = self.momentum
        epoch = 0
        epoch_error = self.error_tolerance
        epoch_errors = []
        epoch_size = len(norm_inputs)
        while epoch < self.max_epoch and epoch_error >= self.error_tolerance:
            epoch += 1
            epoch_error = 0
            for i in range(epoch_size):
                output = self.feed_forward(norm_inputs[i])
                error = output[-1] - norm_labels[i]
                self.nabla_biases, self.nabla_weights = self._back_propagation(error, output)
                self.biases -= momentum * self.nabla_biases
                self.weights -= momentum * self.nabla_weights
                epoch_error += 0.5 * sum(error * error)
            epoch_error /= epoch_size
            epoch_errors.append(epoch_error)
        return numpy.array(epoch_errors)


if __name__ == "__main__":
    figure, axis = pyplot.subplots()
    sample_inputs = numpy.array([[0.1], [0.2], [0.3], [0.4], [0.5], [0.7], [0.8], [0.9], [1.0]])
    sample_labels = numpy.array([[0.001], [0.004], [0.009], [0.016], [0.025], [0.049], [0.064], [0.81], [0.100]])
    test_input = numpy.array([0.6])
    test_label = numpy.array([0.036])
    network = FullConnectedNetwork([1, 2, 3, 2, 1])
    start_time = time.time()
    training_errors = network.stochastic_gradient_descent_training(sample_inputs, sample_labels)
    finish_time = time.time()
    period = finish_time - start_time
    print("Time consumed for training is {} secs with training error {}".format(period, training_errors[-1]))
    axis.plot(training_errors, color="green")
    pyplot.show()
