from math import e
from multiprocessing import Manager, Pool, cpu_count
from numpy import array, mean, var
from numpy.random import normal


def evolution_strategy(features, fitness, population_size, elite_size, stopping_check=lambda x: False, total_epoch=100):

    """Canonical Evolution Strategy"""

    with Manager() as manager, Pool(processes=cpu_count()) as pool:
        for epoch in range(total_epoch):
            if epoch == 0:
                population = manager.list(array([normal(f[0], f[1], population_size) for f in features]).T)
            else:
                population = manager.list(array([normal(m, s, population_size) for m, s in zip(means, scales)]).T)
            fitness_values = pool.map(fitness, population)
            results = zip(population, fitness_values)
            ordered_results = sorted(results, key=lambda x: x[1], reverse=True)
            if stopping_check(ordered_results[0][1]):
                break
            else:
                successors = array(ordered_results)[:, 0][:elite_size]
                means = mean(successors)
                scales = var(successors)
        return population[0]


if __name__ == "__main__":
    def fitness(params):
        return 1 + params[0] + params[1]

    features = array([
        [0.7, 0.2],
        [1.2, 0.5]
    ])

    answers = evolution_strategy(features, fitness, 10, 5)
    print(answers)
