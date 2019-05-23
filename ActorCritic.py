import matplotlib.pyplot as plot
from numpy import argmax, array, dot, ndarray, where, zeros
from numpy.random import choice, random


def _take_action(π):
    pick = random()
    i = -1
    while pick >= 0:
        i += 1
        pick -= π[i]
    return i


def one_step_actor_critic(**kwargs):

    """One-step Actor Critic algorithm"""

    S = kwargs.pop("states")
    if not isinstance(S, ndarray) or not isinstance(S.size, int):
        raise Exception("states are needed to be one dimensional numpy.ndarray object")
    NS = S.size

    T = kwargs.pop("terminate_state")
    if T not in S:
        raise Exception("terminal state is not one of the state")
    t, = where(S == T)

    A = kwargs.pop("actions")
    if not isinstance(A, ndarray) or not isinstance(A.size, int):
        raise Exception("actions are needed to be one dimensional numpy.ndarray object")
    NA = A.size

    Nθ = kwargs.pop("policy_parameter_space")
    if not isinstance(Nθ, int):
        raise Exception("policy_parameter_space is needed to be int typed")
    θ = random(Nθ)

    NW = kwargs.pop("weight_size")
    if not isinstance(NW, int):
        raise Exception("weight size should be int typed")
    W = random(NW)

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
        aθ = kwargs.pop("policy_alpha")
    except:
        aθ = 0.8
    if not isinstance(aθ, float):
        raise Exception("policy_alpha must be float typed")

    try:
        aw = kwargs.pop("value_alpha")
    except:
        aw = 0.8
    if not isinstance(aw, float):
        raise Exception("value_alpha must be float typed")

    try:
        γ = kwargs.pop("gamma")
    except:
        γ = 0.8
    if not isinstance(γ, float):
        raise Exception("gamma must be integer typed")

    try:
        episode_size = kwargs.pop("episode_size")
    except:
        episode_size = 100
    if not isinstance(episode_size, int):
        raise Exception("episode size must be int typed")

    V = random([NS, NW])
    π = random([NS, NA, Nθ])

    td_errors = []

    for episode in range(episode_size):

        s = choice(range(NS))

        while s != t[0]:
            a = _take_action([max([dot(a, θ), 1]) for a in π[s]])
            s_prime = argmax(P[s, a])
            r_prime = R[s, a, s_prime]
            td_error = r_prime + γ * dot(V[s_prime], W) - dot(V[s], W)
            # W ← W + α * [􏰄R + γV(s′) − V(s)􏰅] * ∇V(s)
            W = W + aw * td_error * V[s]
            θ = θ + aθ * td_error * (π[s, a] / dot(π[s, a], θ))
            s = s_prime

        try:
            td_errors.append(td_error)
        except:
            print("No TD error is found as the first step of this episode is terminal state.")

    plot.plot(td_errors)
    plot.show()


def one_step_actor_critic_with_eligibility_trace(**kwargs):

    """One-step Actor Critic algorithm with eligibility trace optimization"""

    S = kwargs.pop("states")
    if not isinstance(S, ndarray) or not isinstance(S.size, int):
        raise Exception("states are needed to be one dimensional numpy.ndarray object")
    NS = S.size

    T = kwargs.pop("terminate_state")
    if T not in S:
        raise Exception("terminal state is not one of the state")
    t, = where(S == T)

    A = kwargs.pop("actions")
    if not isinstance(A, ndarray) or not isinstance(A.size, int):
        raise Exception("actions are needed to be one dimensional numpy.ndarray object")
    NA = A.size

    Nθ = kwargs.pop("policy_parameter_space")
    if not isinstance(Nθ, int):
        raise Exception("policy_parameter_space is needed to be int typed")
    θ = random(Nθ)

    NW = kwargs.pop("weight_size")
    if not isinstance(NW, int):
        raise Exception("weight size should be int typed")
    W = random(NW)

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
        aθ = kwargs.pop("policy_alpha")
    except:
        aθ = 0.8
    if not isinstance(aθ, float):
        raise Exception("policy_alpha must be float typed")

    try:
        aw = kwargs.pop("value_alpha")
    except:
        aw = 0.8
    if not isinstance(aw, float):
        raise Exception("value_alpha must be float typed")

    try:
        γ = kwargs.pop("gamma")
    except:
        γ = 0.8
    if not isinstance(γ, float):
        raise Exception("gamma must be integer typed")

    try:
        episode_size = kwargs.pop("episode_size")
    except:
        episode_size = 100
    if not isinstance(episode_size, int):
        raise Exception("episode size must be int typed")

    V = random([NS, NW])
    π = random([NS, NA, Nθ])

    td_errors = []

    for episode in range(episode_size):

        s = choice(range(NS))
        Ew = zeros([NS, NW])
        Eθ = zeros([NS, NA, Nθ])

        while s != t[0]:
            a = _take_action([max([dot(a, θ), 1]) for a in π[s]])
            s_prime = argmax(P[s, a])
            r_prime = R[s, a, s_prime]
            td_error = r_prime + γ * dot(V[s_prime], W) - dot(V[s], W)
            Ew[s] = Ew[s] + 1
            Eθ[s, a] = Eθ[s, a] + 1
            # W ← W + α * [􏰄R + γV(s′) − V(s)􏰅] * ∇V(s)
            W = W + aw * td_error * V[s] * Ew[s]
            θ = θ + aθ * td_error * (π[s, a] / dot(π[s, a], θ)) * Eθ[s, a]
            Ew = γ * Ew
            Eθ = γ * Eθ
            s = s_prime

        try:
            td_errors.append(td_error)
        except:
            print("No TD error is found as the first step of this episode is terminal state.")

    plot.plot(td_errors)
    plot.show()


if __name__ == "__main__":
    S = array(["s1", "s2", "s3", "s4", "s5"])
    A = array(["a1", "a2", "a3", "a4"])
    one_step_actor_critic(states=S, actions=A, terminate_state="s2", policy_parameter_space=4, weight_size=3)
    one_step_actor_critic_with_eligibility_trace(
        states=S,
        actions=A,
        terminate_state="s2",
        policy_parameter_space=4,
        weight_size=3
    )

