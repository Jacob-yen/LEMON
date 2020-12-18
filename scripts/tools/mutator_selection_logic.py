import numpy as np
from scripts.logger.lemon_logger import Logger
import random
np.random.seed(20200501)

class Roulette:
    class Mutant:
        def __init__(self, name, selected=0):
            self.name = name
            self.selected = selected

        @property
        def score(self):
            return 1.0 / (self.selected + 1)

    def __init__(self, mutant_names=None, capacity=101):
        self.capacity = capacity
        if mutant_names is None:
            self._mutants = []
        else:
            self._mutants = [Roulette.Mutant(name) for name in mutant_names]

    @property
    def mutants(self):
        mus = {}
        for mu in self._mutants:
            mus[mu.name] = mu
        return mus

    @property
    def pool_size(self):
        return len(self._mutants)

    def add_mutant(self, mutant_name, total=0):
        self._mutants.append(Roulette.Mutant(mutant_name, total))

    def pop_one_mutant(self, ):
        np.random.shuffle(self._mutants)
        self._mutants.pop()

    def is_full(self):
        if len(self._mutants) >= self.capacity:
            return True
        else:
            return False

    def choose_mutant(self):
        sum = 0
        for mutant in self._mutants:
            sum += mutant.score
        rand_num = np.random.rand() * sum
        for mutant in self._mutants:
            if rand_num < mutant.score:
                return mutant.name
            else:
                rand_num -= mutant.score


class MCMC:
    class Mutator:
        def __init__(self, name, total=0, delta_bigger_than_zero=0, epsilon=1e-7):
            self.name = name
            self.total = total
            self.delta_bigger_than_zero = delta_bigger_than_zero
            self.epsilon = epsilon

        @property
        def score(self, epsilon=1e-7):
            mylogger = Logger()
            rate = self.delta_bigger_than_zero / (self.total + epsilon)
            mylogger.info("Name:{}, rate:{}".format(self.name, rate))
            return rate

    def __init__(self, mutate_ops=None):
        self.mylogger = Logger()
        self.mylogger.info(f"Using {self.__class__.__name__} as selection strategy!")
        if mutate_ops is None:
            from scripts.mutation.model_mutation_generators import all_mutate_ops
            mutate_ops = all_mutate_ops()
        self.p = 1 / len(mutate_ops)
        self._mutators = [self.Mutator(name=op) for op in mutate_ops]

    @property
    def mutators(self):
        mus = {}
        for mu in self._mutators:
            mus[mu.name] = mu
        return mus

    def choose_mutator(self, mu1=None):
        if mu1 is None:
            # which means it's the first mutation
            return self._mutators[np.random.randint(0, len(self._mutators))].name
        else:
            self.sort_mutators()
            k1 = self.index(mu1)
            k2 = -1
            prob = 0
            while np.random.rand() >= prob:
                k2 = np.random.randint(0, len(self._mutators))
                prob = (1 - self.p) ** (k2 - k1)
            mu2 = self._mutators[k2]
            return mu2.name

    def sort_mutators(self):
        import random
        random.shuffle(self._mutators)
        self._mutators.sort(key=lambda mutator: mutator.score, reverse=True)

    def index(self, mutator_name):
        for i, mu in enumerate(self._mutators):
            if mu.name == mutator_name:
                return i
        return -1