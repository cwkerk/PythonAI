from numpy import argmax, array, ndarray, ones
from numpy.random import random


def epsilon_greedy(**kwargs):
    A = kwargs.pop("actions")
    if not isinstance(A, ndarray):
        raise Exception("actions are needed to be one dimensional numpy.ndarray object")
    NA = A.size
    if not isinstance(NA, int):
        raise Exception("actions are needed to be one dimensional numpy.ndarray object")

    Q = kwargs.pop("action_values")
    if not isinstance(Q, ndarray) or not isinstance(Q.shape, tuple) or len(Q.shape) != 2 or Q.shape[1] != NA:
        raise Exception("action values are needed to be two dimensional numpy.ndarray object")

    try:
        e = kwargs.pop("epsilon")
    except:
        e = 0.1
    if not isinstance(e, float):
        raise Exception("epsilon must be float typed")

    i = kwargs.pop("state_index")
    if not isinstance(i, int) or i >= Q.shape[0]:
        raise Exception("state_index is needed to be the index of interested state in state space")

    # ∀ a, π(a | s) = ε / |A(s)|
    π = ones(NA, dtype=float) * e / NA
    best_action = argmax(Q[i])
    # π(a• | s) =  1 - ε + (ε / |A(s)|)
    π[best_action] += (1.0 - e)
    return π


if __name__ == "__main__":
    states = array(["s1", "s2", "s3", "s4", "s5"])
    actions = array(["a1", "a2", "a3"])
    action_values = random([states.size, actions.size])
    policy = epsilon_greedy(actions=actions, action_values=action_values, state_index=2)
    print(policy)
