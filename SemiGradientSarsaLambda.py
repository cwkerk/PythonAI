from numpy import argmax, array, dot, ones, where, zeros
from numpy.random import choice, random


class Environment:

    def __init__(self, **kwargs):
        self.S = kwargs.pop("states")
        self.S_T = kwargs.pop("terminate_state")
        self.A = kwargs.pop("actions")
        self.NS = self.S.size
        self.NA = self.A.size
        self.P = random([self.NS, self.NA, self.NS])


class Agent:

    @staticmethod
    def _take_action(π):
        pick = random()
        i = -1
        while pick >= 0:
            i += 1
            pick -= π[i]
        return i

    _alpha_ = 0.8
    _epsilon_ = 0.1
    _gamma_ = 0.7
    _lambda_ = 0.5

    def __init__(self, **kwargs):
        self.environment = kwargs.pop("environment")
        self.Q = zeros([self.environment.NS, self.environment.NA, 3])
        self.R = random([self.environment.NS, self.environment.NA, self.environment.NS])
        self.W = random(3)

    def _epsilon_greedy(self, si):
        π = ones(self.environment.NA) * self._epsilon_ / self.environment.NA
        a_best = argmax([self._get_value_approximator(si, ai) for ai in range(self.environment.NA)])
        π[a_best] += (1 - self._epsilon_)
        return π

    def _get_value_approximator(self, si, ai):
        return dot(self.W, self.Q[si, ai])

    def _get_value_approximator_gradient(self, si, ai):
        return sum(self.Q[si, ai])

    def on_policy_training(self, episode_size=100):
        S = self.environment.S
        S_T = self.environment.S_T
        P = self.environment.P
        s_t, = where(S == S_T)
        for episode in range(episode_size):
            print("{}th episode starts".format(episode))
            s = choice(self.environment.NS)
            z = 0
            while s != s_t[0]:
                π = self._epsilon_greedy(s)
                a = self._take_action(π)
                s_prime = argmax(P[s, a])
                π_prime = self._epsilon_greedy(s_prime)
                a_prime = self._take_action(π_prime)
                delta = self._gamma_ + self._get_value_approximator_gradient(s, a)
                z = self._gamma_ * self._lambda_ * z + delta
                td_error = self.R[s, a, s_prime] + self._gamma_ * self.Q[s_prime, a_prime] - self.Q[s, a]
                self.Q[s, a] = self.Q[s, a] + self._alpha_ * td_error
                self.W = self.W + self._alpha_ * td_error * z
                s = s_prime
                print("{}th: {}".format(episode, td_error))
            print("{}th episode ends".format(episode))


if __name__ == "__main__":
    states = array(["s1", "s2", "s3", "s4", "s5"])
    actions = array(["a1", "a2", "a3"])
    environment = Environment(states=states, actions=actions, terminate_state="s4")
    agent = Agent(environment=environment)
    print("---------------------------------------------")
    print("Initial Q: {}".format(agent.Q))
    print("---------------------------------------------")
    print("Initial W: {}".format(agent.W))
    print("---------------------------------------------")
    agent.on_policy_training()
    print("---------------------------------------------")
    print("Final Q: {}".format(agent.Q))
    print("---------------------------------------------")
    print("Final W: {}".format(agent.W))
