from numpy import argmax, array, dot, ones, where
from numpy.random import choice, random


NW = 3
# W, weight vector
W = random(NW)


class Environment:

    def __init__(self, **kwargs):
        self.S = kwargs.pop("states")
        self.A = kwargs.pop("actions")
        self.S_T = kwargs.pop("terminate_state")
        self.NS = self.S.size
        self.NA = self.A.size
        self.P = random([self.NS, self.NA, self.NS])
        self.Q = random([self.NS, self.NA, NW])
        self.R = random([self.NS, self.NA, self.NS])


class Agent:

    W = random(3)

    @staticmethod
    def _take_action(π):
        pick = random()
        i = -1
        while pick >= 0:
            i += 1
            pick -= π[i]
        return i

    alpha = 0.8
    epsilon = 0.1
    gamma = 0.7

    def __init__(self, **kwargs):
        self.environment = kwargs.pop("environment")

    def _epsilon_greedy(self, si):
        π = ones(self.environment.NA) * self.epsilon / self.environment.NA
        a_best = argmax([self._get_value_approximator(si, ai) for ai in range(self.environment.NA)])
        π[a_best] += (1 - self.epsilon)
        return π

    def _get_value_approximator(self, si, ai):
        return dot(self.W, self.environment.Q[si, ai])

    def _get_value_approximator_gradient(self, si, ai):
        return sum(self.environment.Q[si, ai])

    def on_policy_training(self, episode_size=100):
        S = self.environment.S
        S_T = self.environment.S_T
        P = self.environment.P
        R = self.environment.R
        s_t, = where(S == S_T)
        for episode in range(episode_size):
            s = choice(self.environment.NS)
            while s != s_t[0]:
                Q = self.environment.Q
                π = self._epsilon_greedy(s)
                a = self._take_action(π)
                s_prime = argmax(P[s, a])
                π_prime = self._epsilon_greedy(s_prime)
                a_prime = self._take_action(π_prime)
                td_error = R[s, a, s_prime] + self.gamma * Q[s_prime, a_prime] - Q[s, a]
                self.environment.Q[s, a] = Q[s, a] + self.alpha * td_error
                self.W = self.W + self.alpha * td_error * self._get_value_approximator_gradient(s, a)
                s = s_prime
                print("{}th: {}".format(episode, td_error))


if __name__ == "__main__":
    states = array(["s1", "s2", "s3", "s4", "s5"])
    actions = array(["a1", "a2", "a3"])
    environment = Environment(states=states, actions=actions, terminate_state="s4")
    agent = Agent(environment=environment)
    print("---------------------------------------------")
    print("Initial Q: {}".format(environment.Q))
    print("---------------------------------------------")
    print("Initial W: {}".format(W))
    print("---------------------------------------------")
    agent.on_policy_training()
    print("---------------------------------------------")
    print("Final Q: {}".format(environment.Q))
    print("---------------------------------------------")
    print("Final W: {}".format(W))
