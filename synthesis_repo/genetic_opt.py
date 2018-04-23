import numpy as np

class BaseGenetics(object):
    def __init__(self, e=None):
        self.estimator = None
        self.expectations = e # vector of math expectations for each component
        
    def set_estimator(self, f):
        def g(*args, **kwargs):
            return f(*args, **kwargs)
        self.estimator = g
        
    def generate_population(self, qmin, qmax, h, m):
        """
        Generate population.
        
        (real, real, int, int) -> [h x m] np.array of reals
        """
        population = {}
        e = self.expectations
        if e:
            functional = self.estimate_object_function(e)
            population[functional] = e
            while len(population) < h:
                candidate = np.random.normal(e, 0.03)
                functional = self.estimate_object_function(candidate)
                if functional < 1e+3: population[functional] = candidate
        else:
            while len(population) < h:
                candidate = np.random.uniform(qmin, qmax, m)
                functional = self.estimate_object_function(candidate)
                if functional < 1e+3: population[functional] = candidate
        return population
    
    def estimate_object_function(self, q):
        """
        Evaluates function self.estimator with q as an incoming parameter
        
        (vector) -> real
        """
        return self.estimator(q)

    
    def get_best_individual(self, population, worst=False, ksearch=None):
        """
        Return best or worst individual:
        1) if ksearch != None and worst==False: return best individual
        from ksearch random sample without replacement.
        2) if ksearch == None and worst==True: return index of the worst
        individual from the whole population.

        (2d array of real, bool, int) -> array of real OR int
        """
        population_estimates = np.array(list(population.keys()))
        if ksearch and not worst:
            try:
                subpopulation_estimates = population_estimates[np.random.choice(population_estimates.shape[0], ksearch, replace=False)]
                individual_estimate = subpopulation_estimates.min()
                return (population[individual_estimate], individual_estimate)
            except ValueError as e: print('Wrong type for ksearch: {0}'.format(e))
        else:
            best_estimate = population_estimates.min()
            return (population[best_estimate], best_estimate)
    
    def cross(self, population, ksearch):
        """
        Processes crossover of some individuals.

        (array of array of reals, int) -> (array of real, array of real) OR None
        """
        best_individual, best_value = self.get_best_individual(population)
        if len(best_individual) > 1:
            parent1, parent1_est = self.get_best_individual(population, worst=False, ksearch=ksearch)
            parent2, parent2_est = self.get_best_individual(population, worst=False, ksearch=ksearch)
            if np.max([best_value/parent1_est, best_value/parent2_est])>np.random.uniform():
                crossover_point = np.random.randint(1, len(parent1) - 1)
                child1 = np.hstack((parent1[:crossover_point], parent2[crossover_point:]))
                child2 = np.hstack((parent2[:crossover_point], parent1[crossover_point:]))
                return (child1, child2)
            else: return None
        elif len(best_individual) == 1: return (best_individual[:], best_individual[:])
        else: print('fuck you')
        
    def mutate(self, children, qmin, qmax, p=1):
        """
        Mutate given child1 and child2 with probability 'p'.

        (array of real, array of real, real, real, real) -> None
        """
        if np.random.rand() < p:
            mutated_children = {}
            for child in children:
                child_gene = np.random.randint(child.shape[0])
                child[child_gene] = np.random.uniform(qmin, qmax)
                child_functional = self.estimate_object_function(child)
                mutated_children[child_functional] = child
            return mutated_children
        else: return None
        
    def insert_children(self, population, children):
        """
        Replace the worst individuals with children, if they fit better.

        (2d array of real, array of real, array of real) -> None
        """
        merge = {**children, **population}
        k = len(children)
        estimates = list(merge.keys()) # unique estimates
        bad_k = np.partition(estimates, k)[-k:]
        for e in bad_k: del merge[e]
        return merge
                
    # psi_change_epoch <= individuals
    # ksearch <= individuals
    # variations_per_individuals >= 1
    # g > 0
    # crossings > 0
    
    def optimize(self, qmin=1, qmax=4, individuals=1000, generations=10,
                       individual_len=3, crossings=256, ksearch=16):
        print('Generating population for parametric optimization...')
        population = self.generate_population(qmin, qmax, individuals, individual_len)
        for g in range(generations):
            for c in range(crossings):
                children = self.cross(population, ksearch)
                if children:
                    children = self.mutate(children, qmin, qmax)
                    population = self.insert_children(population, children)
                    if len(population) <= ksearch:
                        print('Population died out!')
                        best_individual, best_value = self.get_best_individual(population)
                        print('J: {0}, Q: {1}'.format(best_value, best_individual))
                        return best_individual, best_value
                else: continue
            best_individual, best_value = self.get_best_individual(population)
            print('J: {0}, Q: {1}'.format(best_value, best_individual))
        return best_individual, best_value

if __name__=='__main__':
    estimator = lambda q: np.linalg.norm(np.array([1,2,3]) - q)
    go = BaseGenetics()
    go.set_estimator(estimator)
    go.optimize()

