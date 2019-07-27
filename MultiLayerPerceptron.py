from matplotlib import pyplot
from numpy import array, dot, exp, outer, random, zeros


def _cost_derivative(output, target):
    return output - target


def sigmoid(z):
    return 1 / (1 + exp(-z))


def sigmoid_derivative(a):
    return a * (1 - a)


class Network:
    max_epoch = 500000000000
    threshold = 1e-4

    def __init__(self, sizes):
        # TODO: type check for `sizes`
        self.sizes = sizes
        # create biases for hidden and output nodes
        self.biases = [random.randn(size) for size in sizes[1:]]
        self.weights = [random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    # both expected_outputs is data list of size same as output layer
    def _back_propagation(self, targets, outputs):
        nabla_biases = [zeros(bias.shape) for bias in self.biases]
        nabla_weights = [zeros(weight.shape) for weight in self.weights]
        delta = _cost_derivative(outputs[-1], targets) * sigmoid_derivative(outputs[-1])
        # cost to correct output bias,
        # ∂(error)/∂(output_bias) = ∂(error)/∂(output)
        nabla_biases[-1] = delta
        # cost to correct output weight,
        # ∂(error)/∆(output_weight)
        #   = ∂(error)/∂(output) * ∂(output)/∂(weight)
        #   = outer product of ∂(error)/∂(output) & ∂(output)/∂(weight)
        nabla_weights[-1] = outer(delta, outputs[-2])
        # From second last layer to first layer
        for i in range(2, len(self.sizes)):
            # compute delta,
            # delta
            #   = ∂(error)/∂(curr_output)
            #   = ∂(error)/∂(last_output) * ∂(last_output)/∂(curr_activation) * ∂(curr_activation)/∂(curr_output)
            #   = delta * weight * sigmoid_derivative(current_output)
            delta = dot(delta, self.weights[-i + 1]) * sigmoid_derivative(outputs[-i])
            nabla_biases[-i] = delta
            nabla_weights[-i] = outer(delta, outputs[-i - 1])
        return nabla_biases, nabla_weights

    def _stochastic_gradient_descent(self, targets, outputs, learning_rate):
        nabla_biases, nabla_weights = self._back_propagation(targets, outputs)
        self.biases = [b - learning_rate * nb for b, nb in zip(self.biases, nabla_biases)]
        self.weights = [w - learning_rate * nw for w, nw in zip(self.weights, nabla_weights)]

    def feed_forward(self, input):
        a = input
        outputs = []
        activations = [a]
        for (b, w) in zip(self.biases, self.weights):
            z = dot(w, a) + b
            outputs.append(z)
            a = sigmoid(z)
            activations.append(a)
        return activations

    def train_with_stochastic_gradient_descent(self, training_labels, training_inputs, learning_rate=0.8):
        # TODO: type check for all input arguments
        epoch = 0
        epoch_error = self.threshold
        training_errors = []
        label_size = len(training_inputs)
        inputs = [sigmoid(x) for x in training_inputs]
        targets = [sigmoid(y) for y in training_labels]
        while epoch <= self.max_epoch and epoch_error >= self.threshold:
            epoch += 1
            outputs = []
            for i in range(label_size):
                outputs.append(self.feed_forward(inputs[i]))
                training_error = 0.5 * sum([abs(e - o) ** 2 for e, o in zip(targets[i], outputs[i][-1])])
                self._stochastic_gradient_descent(targets[i], outputs[i], learning_rate)
                epoch_error += training_error
            training_errors.append(epoch_error)
        return training_errors


# As an example:
if __name__ == "__main__":
    figure, axis = pyplot.subplots()
    sample_inputs = array([[0.5], [0.8]])
    sample_labels = array([[0.025], [0.064]])
    net = Network([1, 3, 4, 5, 6, 5, 4, 3, 1])
    errors = net.train_with_stochastic_gradient_descent(sample_labels, sample_inputs)
    axis.plot(errors, color="green")
    errors_2 = net.train_with_stochastic_gradient_descent(sample_labels, sample_inputs, learning_rate=0.6)
    axis.plot(errors_2, color="blue")
    errors_3 = net.train_with_stochastic_gradient_descent(sample_labels, sample_inputs, learning_rate=0.4)
    axis.plot(errors_3, color="red")
    print("Final Error: {}".format(errors[-1]))
    print("Predict output given input 0.6: {}".format(net.feed_forward([0.6])[-1]))
    pyplot.show()
