from random import randrange, sample


def _binary_to_int(text):
    return int(text, 2)


def _int_to_binary(num, digit):
    return "{0:b}".format(digit)


class Chromosome:
    fitness = None
    genes = []

    def __init__(self, genes):
        self.genes = genes

    def mutate(self, gene_set):
        index = randrange(0, len(self.genes))
        genes = list(gene_set).copy()
        genes.remove(self.genes[index])
        mutant = sample(genes, 1)
        self.genes[index] = mutant[0]


class Population:
    best_chromosome = None
    chromosomes = []
    max_epoch = 1000
    seek_for_max = True

    @staticmethod
    def fitness_comparator(x, y):
        return max(x, y)

    def __init__(self, size, gene_set, gene_size, fitness_evaluator):
        # TODO: type check for `fitness_evaluator`, `gene_set`, `gene_size` and `size`
        # `gene_set` represents the variety of gene components.
        # Ex. if gene_set = ["a", "b"], then the "genes" hold by each Chromosome must be combination of "a" and "b"
        self.gene_set = gene_set
        self.gene_size = gene_size
        self.fitness_evaluator = fitness_evaluator
        self.chromosomes = [Chromosome(self._generate_genes()) for _ in range(0, size)]
        self.fitness = [0 for _ in range(0, size)]

    def _crossover(self):
        children = []
        group_size = int(0.5 * len(self.chromosomes))
        group_a = self.chromosomes[:group_size]
        group_b = self.chromosomes[group_size:]
        for pair in range(0, group_size):
            index = int(0.5 * len(group_a[pair].genes))
            child_genes_a = group_a[pair].genes[:index] + group_b[pair].genes[index:]
            child_genes_b = group_b[pair].genes[:index] + group_a[pair].genes[index:]
            children.append(Chromosome(child_genes_a))
            children.append(Chromosome(child_genes_b))
        self.chromosomes = self.chromosomes + children

    def _fitness_evaluation(self):
        best_fitness = None
        for chromosome in self.chromosomes:
            chromosome.fitness = self.fitness_evaluator(chromosome.genes)
            if best_fitness is None or self.fitness_comparator(best_fitness, chromosome.fitness) == chromosome.fitness:
                best_fitness = chromosome.fitness
                self.best_chromosome = chromosome

    def _generate_genes(self):
        return sample(self.gene_set, self.gene_size)

    def _mutate(self):
        for chromosome in self.chromosomes:
            if self.best_chromosome is not None and chromosome.fitness != self.best_chromosome.fitness:
                chromosome.mutate(self.gene_set)

    def _selection(self):
        size = int(0.5 * len(self.chromosomes))
        self.chromosomes.sort(key=lambda chromosome: chromosome.fitness)
        if self.seek_for_max:
            self.chromosomes = self.chromosomes[size:]
        else:
            self.chromosomes = self.chromosomes[:size]

    def run(self):
        epoch = 0
        self._fitness_evaluation()
        while epoch < self.max_epoch and len(self.chromosomes) > 1:
            epoch += 1
            self._selection()
            self._crossover()
            self._mutate()
            self._fitness_evaluation()
        return self.best_chromosome
