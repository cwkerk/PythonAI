from numpy import random, subtract


class SearchSpace:

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def create_position(self, dimension):
        return random.uniform(self.lower_bound, self.upper_bound, size=(dimension[0], dimension[1]))[0]


class Particle:
    best_fitness = None
    best_position = None
    fitness = None
    position = None
    velocity = 0

    def __init__(self, position):
        self.position = position
        self.best_position = self.position

    def move(self):
        self.position = self.position + self.velocity


class Swarm:

    # weight of initial velocity in new velocity computation for each particle
    initial_velocity_weight = 0.2
    # weight of toward global best acceleration in new velocity computation for each particle
    toward_global_best_acceleration = 1.9
    # weight of toward particle best acceleration in new velocity computation for each particle
    toward_particle_best_acceleration = 1.9

    max_epoch = 1000
    particles = []
    search_space = SearchSpace(0, 0)

    @staticmethod
    def fitness_comparator(x, y):
        return max(x, y)

    def __init__(self, size, dimension, search_space, fitness_evaluator):
        self.search_space = search_space
        self.fitness_evaluator = fitness_evaluator
        self.particles = [Particle(self.search_space.create_position(dimension)) for _ in range(0, size)]
        for particle in self.particles:
            particle.fitness = self.fitness_evaluator(particle.position)
            particle.best_fitness = particle.fitness
        self.best_fitness = None
        self._compute_global_best()

    def _compute_global_best(self):
        for particle in self.particles:
            if self.best_fitness is None:
                # update best position for the swarm
                self.best_fitness = particle.best_fitness
                self.best_position = particle.best_position
            elif self.fitness_comparator(self.best_fitness, particle.fitness) == particle.fitness:
                # update best position for the swarm
                self.best_fitness = particle.best_fitness
                self.best_position = particle.best_position

    def _compute_particle_best(self):
        for particle in self.particles:
            particle.fitness = self.fitness_evaluator(particle.position)
            if self.fitness_comparator(particle.best_fitness, particle.fitness) == particle.fitness:
                # update best position for individual particle
                particle.best_fitness = particle.fitness
                particle.best_position = particle.position

    def _move(self):
        for particle in self.particles:
            global_acceleration = self.toward_global_best_acceleration * random.random()
            global_distance = subtract(self.best_position, particle.position)
            global_factor = global_acceleration * global_distance
            local_acceleration = self.toward_particle_best_acceleration * random.random()
            local_distance = subtract(particle.best_position, particle.position)
            local_factor = local_acceleration * local_distance
            particle.velocity = self.initial_velocity_weight + global_factor + local_factor
            particle.move()

    def run(self):
        for _ in range(0, self.max_epoch):
            self._compute_particle_best()
            self._compute_global_best()
            self._move()


# As an example:
if __name__ == "__main__":
    space = SearchSpace(-1, 1)
    swarm = Swarm(100, [100, 100], space, lambda pos: min(pos[0], pos[1]))
    swarm.run()
