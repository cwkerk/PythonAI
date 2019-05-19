import matplotlib.pyplot as plot

from numpy import argmax, array, ones, where
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

    _alpha_ = 0.7
    _epsilon_ = 0.1
    _gamma_ = 0.8

    def __init__(self, **kwargs):
        self.environment = kwargs.pop("environment")
        self.Q = random([self.environment.NS, self.environment.NA])
        self.R = random([self.environment.NS, self.environment.NA, self.environment.NS])

    def _epsilon_greedy(self, si):
        # ∀ a, π(a | s) = ε / |A(s)|
        A = ones(self.environment.NA, dtype=float) * self._epsilon_ / self.environment.NA
        best_action = argmax(self.Q[si])
        # π(a• | s) =  1 - ε + (ε / |A(s)|)
        A[best_action] += (1.0 - self._epsilon_)
        return A

    def off_policy_prediction(self, episode_size=100):
        NS = self.environment.NS
        S = self.environment.S
        P = self.environment.P
        s_t, = where(S == self.environment.S_T)
        errors = []
        for episode in range(episode_size):
            s = choice(range(NS))
            while s != s_t[0]:
                π = self._epsilon_greedy(s)
                a = self._take_action(π)
                s_prime = argmax(P[s, a])
                td_error = self.R[s, a, s_prime] + self._gamma_ * max(self.Q[s_prime]) - self.Q[s, a]
                # Q(s, a) ← Q(s, a) + α[􏰄R + γ max[Q(s′, a′)] − Q(s, a)􏰅]
                self.Q[s, a] = self.Q[s, a] + self._alpha_ * td_error
                s = s_prime
            errors.append(td_error)
        plot.plot(errors)
        plot.show()


if __name__ == "__main__":
    S = array(["s1", "s2", "s3", "s4", "s5"])
    A = array(["a1", "a2", "a3", "a4"])
    environment = Environment(states=S, actions=A, terminate_state="s5")
    agent = Agent(environment=environment)
    print("Initial Q: {}".format(agent.Q))
    agent.off_policy_prediction()
    print("Final Q: {}".format(agent.Q))
