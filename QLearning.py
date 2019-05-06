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
        self.Q = random([self.NS, self.NA])
        self.R = random([self.NS, self.NA, self.NS])


class Agent:

    @staticmethod
    def _take_action(π):
        pick = random()
        i = -1
        while pick >= 0:
            i += 1
            pick -= π[i]
        return i

    alpha = 0.7
    epsilon = 0.1
    gamma = 0.8

    def __init__(self, **kwargs):
        self.environment = kwargs.pop("environment")

    def _epsilon_greedy(self, si):
        # ∀ a, π(a | s) = ε / |A(s)|
        A = ones(self.environment.NA, dtype=float) * self.epsilon / self.environment.NA
        best_action = argmax(self.environment.Q[si])
        # π(a• | s) =  1 - ε + (ε / |A(s)|)
        A[best_action] += (1.0 - self.epsilon)
        return A

    def _get_action_based_on_epsilon_greedy_policy(self, state):
        A = self._epsilon_greedy(state)
        pick = random()
        i = -1
        while pick > 0:
            i += 1
            pick -= A[i]
        return i

    def off_policy_prediction(self, episode_size=100):
        NS = self.environment.NS
        S = self.environment.S
        P = self.environment.P
        R = self.environment.R
        s_t, = where(S == self.environment.S_T)
        for episode in range(episode_size):
            s = choice(range(NS))
            while s != s_t[0]:
                Q = self.environment.Q
                π = self._epsilon_greedy(s)
                a = self._take_action(π)
                s_prime = argmax(P[s, a])
                error = R[s, a, s_prime] + self.gamma * max(Q[s_prime]) - Q[s, a]
                # Q(s, a) ← Q(s, a) + α[􏰄R + γ max[Q(s′, a′)] − Q(s, a)􏰅]
                self.environment.Q[s, a] = Q[s, a] + self.alpha * error
                s = s_prime


if __name__ == "__main__":
    S = array(["s1", "s2", "s3", "s4", "s5"])
    A = array(["a1", "a2", "a3", "a4"])
    environment = Environment(states=S, actions=A, terminate_state="s5")
    print("Initial Q: {}".format(environment.Q))
    agent = Agent(environment=environment)
    agent.off_policy_prediction()
    print("Final Q: {}".format(environment.Q))
