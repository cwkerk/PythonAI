from matplotlib import pyplot
from numpy import array, dot, exp, outer, random, zeros


def _cost_derivative(output, expected):
    return output - expected


def sigmoid(z):
    return 1 / (1 + exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


class Network:
    max_epoch = 500000000000
    threshold = 1e-30

    def __init__(self, sizes):
        # TODO: type check for `sizes`
        self.sizes = sizes
        # create biases for hidden and output nodes
        self.biases = [random.randn(size) for size in sizes[1:]]
        self.weights = [random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    # both expected_outputs is data list of size same as output layer
    def _back_propagation(self, expected_activation, feed_forward_result):
        nabla_biases = [zeros(bias.shape) for bias in self.biases]
        nabla_weights = [zeros(weight.shape) for weight in self.weights]
        activations = feed_forward_result["activations"]
        outputs = feed_forward_result["outputs"]
        delta = _cost_derivative(activations[-1], expected_activation) * sigmoid_derivative(outputs[-1])
        # cost to correct output bias,
        # ∂(error)/∂(output_bias) = ∂(error)/∂(output)
        nabla_biases[-1] = delta
        # cost to correct output weight,
        # ∂(error)/∆(output_weight)
        #   = ∂(error)/∂(output) * ∂(output)/∂(weight)
        #   = outer product of ∂(error)/∂(output) & ∂(output)/∂(weight)
        nabla_weights[-1] = outer(delta, activations[-2])
        # From second last layer to first layer
        for i in range(2, len(self.sizes)):
            # compute delta,
            # delta
            #   = ∂(error)/∂(curr_output)
            #   = ∂(error)/∂(last_output) * ∂(last_output)/∂(curr_activation) * ∂(curr_activation)/∂(curr_output)
            #   = delta * weight * sigmoid_derivative(current_output)
            delta = dot(delta, self.weights[-i + 1]) * sigmoid_derivative(outputs[-i])
            nabla_biases[-i] = delta
            nabla_weights[-i] = outer(delta, activations[-i - 1])
        return nabla_biases, nabla_weights

    def _stochastic_gradient_descent(self, expected_activations, feed_forward_result, learning_rate):
        nabla_biases, nabla_weights = self._back_propagation(expected_activations, feed_forward_result)
        self.biases = [b - learning_rate * nb for b, nb in zip(self.biases, nabla_biases)]
        self.weights = [w - learning_rate * nw for w, nw in zip(self.weights, nabla_weights)]

    def feed_forward(self, input):
        # TODO: type check for `input`
        a = input
        outputs = []
        activations = [a]
        for (b, w) in zip(self.biases, self.weights):
            z = dot(w, a) + b
            outputs.append(z)
            a = sigmoid(z)
            activations.append(a)
        return {"activations": activations, "outputs": outputs}

    def train_with_stochastic_gradient_descent(self, expected_output, training_input, learning_rate=0.85):
        epoch = 0
        errors = []
        expected_activation = sigmoid(expected_output)
        sigmoidized_input = sigmoid(training_input)
        feed_forward_result = self.feed_forward(sigmoidized_input)
        final_activation = feed_forward_result["activations"][-1]
        training_error = 0.5 * sum([abs(e - o) ** 2 for e, o in zip(expected_activation, final_activation)])
        errors.append(training_error)
        while epoch <= self.max_epoch and training_error >= self.threshold:
            epoch += 1
            self._stochastic_gradient_descent(expected_activation, feed_forward_result, learning_rate)
            feed_forward_result = self.feed_forward(sigmoidized_input)
            final_activation = feed_forward_result["activations"][-1]
            training_error = 0.5 * sum([abs(e - o) ** 2 for e, o in zip(expected_activation, final_activation)])
            errors.append(training_error)
        return errors


# As an example:
if __name__ == "__main__":
    figure, axis = pyplot.subplots()
    # train
    input = array([0.5, 0.8])
    output = array([0.025, 0.064])
    net = Network([2, 3, 4, 5, 6, 5, 4, 3, 2])
    errors = net.train_with_stochastic_gradient_descent(output, input)
    axis.plot(errors, color="green")
    errors_2 = net.train_with_stochastic_gradient_descent(output, input, learning_rate=1.0)
    axis.plot(errors_2, color="blue")
    errors_3 = net.train_with_stochastic_gradient_descent(output, input, learning_rate=1.15)
    print("Final Error: {}".format(errors[-1]))
    pyplot.show()
    # test
    input_2 = array([0.7, 0.2])
    result = net.feed_forward(input_2)
    output_2 = result["outputs"][-1]
    print(output_2)
