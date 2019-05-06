from matplotlib import pyplot
from numpy import array, dot, exp, sum
from numpy.random import rand, uniform


def sigmoid(z):
    return 1 / (1 + exp(-z))


def soft_max(x):
    return exp(x) / sum(exp(x))


class Network:
    max_epoch = 10000

    def __init__(self, visible_nodes_count, hidden_nodes_count):
        # TODO: typecheck for `visible_nodes_count` and `hidden_nodes_count`
        self.hidden_nodes_count = hidden_nodes_count
        self.visible_nodes_count = visible_nodes_count
        self.weights = uniform(-1, 1, size=(visible_nodes_count, hidden_nodes_count))

    def train(self, data, learning_rate=0.1):
        # TODO: typecheck for `data`
        data_sample_count = data.shape[0]
        errors = []
        for epoch in range(0, self.max_epoch):
            #
            # Positive Contrastive Divergence (CD):
            # This process computes `hidden_states_positive` based on input data given.
            # Then, compute associations = outer_product(V, H) where V = visible node states, H = hidden node states.
            #
            hidden_probabilities_positive = array(sigmoid(dot(data, self.weights)))
            hidden_states_positive = hidden_probabilities_positive > rand(data_sample_count, self.hidden_nodes_count)
            associations_positive = dot(data.T, hidden_probabilities_positive)
            #
            # Negative Contrastive Divergence (CD):
            # This process computes `visible_probabilities_negative` based on computed `hidden_states_positive` and
            # and weights.
            # Computed `visible_probabilities_negative` represents the ideal `visible_states` to be given computed
            # `hidden_states_positive` and weights.
            # `hidden_states_negative` will be computed based `visible_probabilities_negative` instead of input data
            # given.
            # Then, compute associations = outer_product(V, H) where V = visible node states, H = hidden node states.
            #
            visible_probabilities_negative = array(sigmoid(dot(hidden_states_positive, self.weights.T)))
            hidden_probabilities_negative = array(sigmoid(dot(visible_probabilities_negative, self.weights)))
            associations_negative = dot(visible_probabilities_negative.T, hidden_probabilities_negative)
            #
            # Learning:
            # Based on differences between associations (positive and negative), the delta for weights to be learn from
            # errors can be computed by following:
            #
            associations_diff = associations_positive - associations_negative
            self.weights += learning_rate*(associations_diff / data_sample_count)
            error = sum((data - visible_probabilities_negative) ** 2)
            errors.append(error)
        return errors

    def run_hidden(self, data):
        # TODO: typecheck for `data` as `data` must be 1xN array
        visible_probabilities = sigmoid(dot(data, self.weights.T))
        visible_states = visible_probabilities > rand(1, self.visible_nodes_count)
        return visible_states

    def run_visible(self, data):
        # TODO: typecheck for `data` as `data` must be 1xN array
        hidden_probabilities = sigmoid(dot(data.T, self.weights))
        hidden_states = hidden_probabilities > rand(1, self.hidden_nodes_count)
        return hidden_states


if __name__ == "__main__":
    training_data = array([
        [1, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 1],
        [1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0],
    ])
    test_data = array([1, 1, 1, 0, 0, 0])
    net = Network(6, 8)
    errors = net.train(training_data)
    pyplot.plot(errors)
    hidden = net.run_visible(test_data)
    print(hidden)
    result = net.run_hidden(hidden)
    print(result)
    error = sum((test_data - result) ** 2)
    print(error)
    pyplot.show()
