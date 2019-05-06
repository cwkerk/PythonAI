from numpy import argmax, array, dot, zeros
from numpy.random import choice, random


class Environment:

    P = array([])
    R = array([])

    def __init__(self, states, actions):
        # TODO: type check for `states` and `actions` to make sure that both of them are array.
        self.state_count = states.shape[0]
        self.action_count = actions.shape[0]
        self.actions = actions
        # P(s' | s, a)
        self.P = random([self.state_count, self.action_count, self.state_count])
        # R(s, a, s')
        self.R = random([self.state_count, self.action_count, self.state_count])


class Agent:

    # discount ratio for learning
    gamma = 1.0

    def __init__(self, environment):
        self.environment = environment

    def policy_evaluation(self, policy):

        """ Policy evaluation to compute value for all states following policy """

        P = self.environment.P
        R = self.environment.R
        NS = self.environment.state_count

        # State value, V <- 0
        V = zeros(NS)

        # ∀ s ∈ S:
        for s in range(NS):
            a = policy[s]
            # V[s] = ∑s′∈S P(s′ | s, π[s])[R(s′ | s, π(s)) + γV(s′)]
            V[s] = dot(P[s, a], (R[s, a] + self.gamma * V))

        return V

    def _policy_improvement(self, policy, V):

        """ Policy improvement to compute policy based on state values computed """

        P = self.environment.P
        R = self.environment.R
        NS = self.environment.state_count
        NA = self.environment.action_count

        policy_not_stable = False

        # ∀ s ∈ S:
        for s in range(NS):

            # Q = ∑a∈A ∑s′∈S P(s′ | s, a)[R(s′ | s, a) + γV(s′)]
            Q = array([dot(P[s, a], (R[s, a] + self.gamma * V)) for a in range(NA)])
            # q* <- max(Q)
            q_best = max(Q)
            # a* <- argmax(a) Q
            a_best = argmax(Q)

            # If π(s) ≠ a*:
            if a_best != policy[s]:
                policy_not_stable = True
                V[s] = q_best
                policy[s] = a_best

        return policy, policy_not_stable

    def policy_iteration(self):

        """ Policy iteration """

        # π(s) ∈ A(s)
        policy = choice(range(self.environment.action_count), self.environment.state_count)

        policy_not_stable = True

        while policy_not_stable:
            V = self.policy_evaluation(policy)
            policy, policy_not_stable = self._policy_improvement(policy, V)

        return self.environment.actions[policy]


# example

if __name__ == "__main__":
    S = array(["s1", "s2", "s3", "s4", "s5"])
    A = array(["a1", "a2", "a3", "a4"])
    env = Environment(S, A)
    agent = Agent(env)
    policy = agent.policy_iteration()
    print(policy)
