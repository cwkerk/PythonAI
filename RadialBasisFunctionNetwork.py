from numpy import array, dot, exp, outer, zeros
from numpy.linalg import norm, pinv
from numpy.random import permutation, random, uniform


def _cost_derivative(output, expected):
    return output - expected


def sigmoid(z):
    return 1 / (1 + exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


class Network:

    max_epoch = 10

    def __init__(self, input_dimension, center_dimension, output_dimension):
        # TODO: typecheck for `input_layer_dimension`, `center_dimension` and `output_layer_dimension`
        self.input_dimension = input_dimension
        self.center_dimension = center_dimension
        self.output_dimension = output_dimension
        self.centers = array([uniform(-1, 1, input_dimension) for _ in range(center_dimension)])
        self.weights = random((center_dimension, output_dimension))
        # TODO: select `beta` based on the input data instead of random number
        self.beta = 8

    def _basis_function(self, center, input):
        # Use Gaussian normal distribution function as basis function
        assert len(input) == self.input_dimension and len(center) == self.input_dimension
        return exp(- self.beta * (norm(input - center) ** 2))

    def _compute_activation(self, inputs):
        assert inputs.shape[1] == self.input_dimension
        input_vector_count = inputs.shape[0]
        # Calculate activations of Radial Basis Function Network:
        #      a ~ N(x, c, ?)
        # where a = activation, x = input vector, h = center of the input vector based on hidden node judgement
        activations = zeros((input_vector_count, self.center_dimension), float)
        for center_index, center in enumerate(self.centers):
            for input_index, input in enumerate(inputs):
                activations[input_index, center_index] = self._basis_function(center, input)
        return activations

    def _stochastic_gradient_descent(self, expected_outputs, outputs, activations, learning_rate=0.85):
        delta_costs = _cost_derivative(outputs, expected_outputs)
        delta_weights = []
        for delta_cost in delta_costs:
            for activation in activations:
                delta_weight = outer(activation.T, delta_cost)
                delta_weights.append(delta_weight)
        for delta_weight in delta_weights:
            self.weights = array([w - learning_rate * dw for w, dw in zip(self.weights, delta_weight)])

    def train(self, inputs, outputs):
        # TODO: typecheck for `inputs` and `outputs`
        input_vector_count = inputs.shape[0]
        indexes = permutation(input_vector_count)[:self.center_dimension]
        # TODO: use K-Means instead of random selection to pick up `centers` of the input for better performance
        self.centers = array([inputs[i, :] for i in indexes])
        activations = self._compute_activation(inputs)
        # Note:
        #   G * W = Y where G = activation vector results from hidden layer, W = weights, Y = output
        # Then,
        #   W = inv(G) * Y
        self.weights = dot(pinv(activations), outputs)
        for epoch in range(self.max_epoch):
            self._stochastic_gradient_descent(outputs, dot(activations, self.weights), activations)

    def test(self, inputs):
        activations = self._compute_activation(inputs)
        return dot(activations, self.weights)


# example
if __name__ == "__main__":
    training_inputs = array([[0.1, 0.2, 0.3], [0.3, 0.4, 0.7], [0.5, 0.6, 1.1]])
    training_outputs = array([[0.05, 0.04], [0.25, 0.16], [0.61, 0.36]])
    net = Network(3, 3, 2)
    net.train(training_inputs, training_outputs)
    result = net.test(array([[0.9, 1.0, 0.2]]))
    print("Result: {}".format(result))
