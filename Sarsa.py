import matplotlib.pyplot as plot
from numpy import argmax, array, ndarray, where, zeros
from numpy.random import choice, random

from EpsilonGreedy import epsilon_greedy


def _take_action(π):
    pick = random()
    i = -1
    while pick >= 0:
        i += 1
        pick -= π[i]
    return i


def sarsa(**kwargs):
    S = kwargs.pop("states")
    if not isinstance(S, ndarray) or not isinstance(S.size, int):
        raise Exception("states are needed to be one dimensional numpy.ndarray object")
    NS = S.size

    T = kwargs.pop("terminate_state")
    if T not in S:
        raise Exception("terminal state is not one of the state")

    A = kwargs.pop("actions")
    if not isinstance(A, ndarray) or not isinstance(A.size, int):
        raise Exception("actions are needed to be one dimensional numpy.ndarray object")
    NA = A.size

    try:
        P = kwargs.pop("transitions")
    except:
        P = random([NS, NA, NS])
    if not isinstance(P, ndarray) or P.shape != tuple([NS, NA, NS]):
        raise Exception("transitions are needed to be numpy.ndarray object with shape of [NS, NA, NS]")

    try:
        R = kwargs.pop("rewards")
    except:
        R = random([NS, NA, NS])
    if not isinstance(R, ndarray) or R.shape != tuple([NS, NA, NS]):
        raise Exception("rewards are needed to be numpy.ndarray object with shape of [NS, NA, NS]")

    try:
        a = kwargs.pop("alpha")
    except:
        a = 0.8
    if not isinstance(a, float):
        raise Exception("alpha must be float typed")

    try:
        e = kwargs.pop("epsilon")
    except:
        e = 0.1
    if not isinstance(e, float):
        raise Exception("epsilon must be float typed")

    try:
        g = kwargs.pop("gamma")
    except:
        g = 0.8
    if not isinstance(e, float):
        raise Exception("gamma must be integer typed")

    try:
        episode_size = kwargs.pop("episode_size")
    except:
        episode_size = 100
    if not isinstance(episode_size, int):
        raise Exception("episode size must be int typed")

    Q = zeros([NS, NA])
    t, = where(S == T)

    td_errors = []

    for episode in range(episode_size):

        s = choice(range(NS))
        π = epsilon_greedy(actions=A, action_values=Q, state_index=int(s))
        a = _take_action(π)

        while s != t[0]:
            s_prime = argmax(P[s, a])
            π_prime = epsilon_greedy(actions=A, action_values=Q, state_index=int(s_prime))
            a_prime = _take_action(π_prime)
            td_error = R[s, a, s_prime] + g * Q[s_prime, a_prime] - Q[s, a]
            # Q(s, a) ← Q(s, a) + α[􏰄R + γQ(s′, a′) − Q(s, a)􏰅]
            Q[s, a] = Q[s, a] + a * td_error
            s = s_prime
            a = a_prime
        td_errors.append(td_error)
    plot.plot(td_errors)
    plot.show()

    return Q


if __name__ == "__main__":
    S = array(["s1", "s2", "s3", "s4", "s5"])
    A = array(["a1", "a2", "a3", "a4"])
    sarsa(states=S, actions=A, terminate_state="s5")
