from matplotlib import pyplot
from numpy import array, dot, exp, outer, random, zeros
import time


def _cost_derivative(output, target):
    return output - target


def sigmoid(z):
    return 1 / (1 + exp(-z))


def sigmoid_derivative(a):
    return a * (1 - a)


class Network:
    max_epoch = 5e10
    threshold = 1e-10

    def __init__(self, sizes):
        # TODO: type check for `sizes`
        self.sizes = sizes
        # create biases for hidden and output nodes
        self.biases = [random.randn(size) for size in sizes[1:]]
        self.weights = [random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.nabla_biases = [zeros(bias.shape) for bias in self.biases]
        self.nabla_weights = [zeros(weight.shape) for weight in self.weights]

    def _back_propagation(self, target, outputs):
        nabla_biases = [zeros(bias.shape) for bias in self.biases]
        nabla_weights = [zeros(weight.shape) for weight in self.weights]
        for i in range(1, len(self.sizes)):
            if i == 1:
                delta = _cost_derivative(outputs[-1], target) * sigmoid_derivative(outputs[-1])
            else:
                # compute delta,
                # delta
                #   = ∂(error)/∂(curr_output)
                #   = ∂(error)/∂(last_output) * ∂(last_output)/∂(curr_activation) * ∂(curr_activation)/∂(curr_output)
                #   = delta * weight * sigmoid_derivative(current_output)
                delta = dot(delta, self.weights[-i+1]) * sigmoid_derivative(outputs[-i])
            # cost to correct output bias,
            # ∂(error)/∂(output_bias) = ∂(error)/∂(output)
            nabla_biases[-i] = delta
            # cost to correct output weight,
            # ∂(error)/∆(output_weight)
            #   = ∂(error)/∂(output) * ∂(output)/∂(weight)
            #   = outer product of ∂(error)/∂(output) & ∂(output)/∂(weight)
            nabla_weights[-i] = outer(delta, outputs[-i-1])
        return nabla_biases, nabla_weights

    def _stochastic_gradient_descent(self, targets, outputs, learning_rate, momentum):
        nabla_biases, nabla_weights = self._back_propagation(targets, outputs)
        self.nabla_biases = [momentum * o - learning_rate * n for o, n in zip(self.nabla_biases, nabla_biases)]
        self.nabla_weights = [momentum * o - learning_rate * n for o, n in zip(self.nabla_weights, nabla_weights)]
        self.biases = [b + nb for b, nb in zip(self.biases, self.nabla_biases)]
        self.weights = [w + nw for w, nw in zip(self.weights, self.nabla_weights)]

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

    def train_with_stochastic_gradient_descent(self, training_labels, training_inputs, learning_rate=1.0, momentum=0.1):
        # TODO: type check for all input arguments
        epoch = 0
        epoch_error = self.threshold
        training_errors = []
        label_size = len(training_inputs)
        inputs = [sigmoid(x) for x in training_inputs]
        targets = [sigmoid(y) for y in training_labels]
        while epoch <= self.max_epoch and epoch_error >= self.threshold:
            epoch += 1
            epoch_error = 0
            for i in range(label_size):
                output = self.feed_forward(inputs[i])
                training_error = 0.5 * sum([abs(e - o) ** 2 for e, o in zip(targets[i], output[-1])])
                self._stochastic_gradient_descent(targets[i], output, learning_rate, momentum)
                epoch_error += training_error
            print(epoch_error)
            training_errors.append(epoch_error)
        return training_errors


# As an example:
if __name__ == "__main__":
    figure, axis = pyplot.subplots()
    sample_inputs = array([[0.1], [0.2], [0.3], [0.4], [0.5], [0.7], [0.8], [0.9], [1.0]])
    sample_labels = array([[0.001], [0.004], [0.009], [0.016], [0.025], [0.049], [0.064], [0.81], [0.100]])
    net = Network([1, 4, 8, 4, 1])
    net.threshold = 1e-5
    training_start_time = time.time()
    errors = net.train_with_stochastic_gradient_descent(sample_labels, sample_inputs)
    training_complete_time = time.time()
    print("Time consumed for SGD training: {} secs".format(training_complete_time - training_start_time))
    axis.plot(errors, color="green")
    print("Prediction output given input 0.6: {}".format(net.feed_forward([0.6])[-1]))
    pyplot.show()
