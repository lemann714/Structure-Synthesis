import random
from operator import itemgetter
import numpy as np
from . import BaseGenetics

class StructureGenetics(object):
    """
    
    """
    def __init__(self, nop, model):
        self.nop = nop
        self.model = model
        self.model.set_control_function(nop.eval_psi)
        self.base_estimation = None
        self.qmin = None
        self.qmax = None
    
    def generate_variations_population(self, individuals, variations_per_individual):
        """
        Generate population.
        
        (int, int) -> np.array of integers (shape: [h x m x 4])
        """
        population = {}
        while len(population) < individuals:
            individual = [self.generate_variation(mutate_param=False) for i in range(variations_per_individual)]
            functional = self.estimate_variation(individual)
            if functional < 1e+3: population[functional] = individual
        return population

    def estimate_object_function(self,):
        """
        Evaluates function self.estimator with q as an incoming parameter
        
        (vector) -> real
        """
        functional = self.model.simulate()
        self.model.reset()
        return functional

    def estimate_variation(self, variation_matrix):
        for variation in variation_matrix:
            if variation[0] != 3: # first we apply all kinds of variations except of the delete
                self.apply_variation(variation)
        for variation in variation_matrix:
            if variation[0] == 3: # then we apply delete variation
                self.apply_variation(variation)
        a = self.nop.get_psi()
        b = self.nop.get_base_psi()
        c = self.nop.get_q()
        d = self.nop.get_base_q()
        if a == b and c == d: est = self.base_estimation
        else: est = self.estimate_object_function()            
        self.nop.roll_back_to_base_psi()
        self.nop.roll_back_to_base_q()
        return est

    ### function to insert into parametric optimization ###
    def estimate_parameters(self, q):
        self.nop.set_q(q)
        functional = self.estimate_object_function()
        return functional
    #######################################################

    def get_best_individual(self, population, ksearch=None):#, worst=False, ksearch=None):
        """
        Return best or worst individual:
        1) if ksearch != None and worst==False: return best individual
        from ksearch random sample without replacement.
        2) if ksearch == None and worst==True: return index of the worst
        individual from the whole population.

        (2d array of real, bool, int) -> array of real OR int
        """
        population_estimates = np.array(list(population.keys()))
        if ksearch:# and not worst:
            try:
                subpopulation_estimates = population_estimates[np.random.choice(population_estimates.shape[0], ksearch, replace=False)]
                individual_estimate = subpopulation_estimates.min()
                return (population[individual_estimate], individual_estimate)
            except ValueError as e: print('Wrong type for ksearch: {0}'.format(e))
        else:
            best_estimate = population_estimates.min()
            return (population[best_estimate], best_estimate)
       
    def generate_variation(self, mutate_param=False):
        psi = self.nop.get_base_psi()
        var_num = random.randint(0,4)
        sublist_index = random.randint(0, len(psi) - 1) # operand column index 
        un_keys_list = self.nop.get_unary_dict_keys()
        bin_keys_list = self.nop.get_binary_dict_keys()
        if var_num == 4 or mutate_param: # nop must have at least one parameter
            param_index = random.randrange(0, self.nop.get_free_input_node())
            new_value = random.uniform(self.qmin, self.qmax)
            if not mutate_param: return [4, param_index, new_value, None]
            else: return [4, mutate_param, new_value, None]
        elif var_num == 0: # change binary operation
            bin_keys_list = self.nop.get_binary_dict_keys()
            new_bin_op = random.choice(bin_keys_list)
            c = random.randint(0, max(un_keys_list[-1], bin_keys_list[-1]))
            return [0, sublist_index, c, new_bin_op]
        elif var_num == 1: # change unary operation
            un_keys_list = self.nop.get_unary_dict_keys()
            l = len(psi[sublist_index])
            unary_cell = random.randint(0, l - 2) # except binary node
            new_un_op = random.choice(un_keys_list)
            return [1, sublist_index, unary_cell, new_un_op]
        elif var_num == 2: # add unary operation
            new_un_op = random.choice(un_keys_list)
            if sublist_index == 0:
                node = random.choice(self.nop.get_input_nodes())
            else:
                node = random.randint(0, psi[sublist_index-1][-1][0])
            return [2, sublist_index, node, new_un_op]
        elif var_num == 3: # delete unary operation
            a = random.randrange(0, len(psi))
            b = random.randrange(0, len(psi[a]))
            c = random.randint(0, max(un_keys_list[-1], bin_keys_list[-1]))
            index_to_start_from_delete = None
            exclude = []
            inputs = self.nop.get_input_nodes()
            for i in inputs:
                for ind, val in enumerate(psi):
                    for j, v in enumerate(val):
                        if v[0] == i:
                            exclude.append((ind, j))
                            break
                    break
                continue
            left_bound = max(exclude, key=itemgetter(0)) # (sublist_index, tuple_index)
            sublist_index = random.randint(left_bound[0], len(psi)-1)
            l = len(psi[sublist_index])
            if l > 3: # if that column has more than one operand
                if sublist_index == left_bound[0]:
                    sample_indices = [j for j, v in enumerate(psi[sublist_index][:-1]) if j != left_bound[1]]
                    if sample_indices:
                        cell_to_del = random.choice(sample_indices)
                    else:
                        return [3, a, b, c]
                else: cell_to_del = random.randint(0, l - 2) # choose random index of the cell, except the last(binary cell)
                node_to_del = psi[sublist_index][cell_to_del][0] # operand row index
                nodes = [list(map(itemgetter(0), sublist[:-1])) for sublist in psi] # all unary nodes (list of lists)
                if sum(x.count(node_to_del) for x in nodes) > 1: return [3, sublist_index, cell_to_del, c] # if more than one occurence 
                else: return [3, a, b, c] # lost graph connectivity
            else: return [3, a, b, c] # lost graph connectivity       

    def apply_variation(self, variation):
        loc_psi = self.nop.get_psi()
        sublist_index = variation[1]
        if variation[0] == 0: # change binary
            new_bin_op = variation[3]
            if new_bin_op > len(self.nop.get_binary_dict_keys()) - 1: return None
            node = loc_psi[sublist_index][-1][0]
            loc_psi[sublist_index][-1] = (node, new_bin_op)
        elif variation[0] == 1: # change unary
            cell = variation[2]
            new_un_op = variation[3]
            if cell >= len(loc_psi[sublist_index]) - 1: return None
            elif new_un_op > len(self.nop.get_unary_dict_keys()) - 1: return None
            node = loc_psi[sublist_index][cell][0]
            loc_psi[sublist_index][cell] = (node, new_un_op)
        elif variation[0] == 2: # add unary
            node = variation[2]
            new_un_op = variation[3]
            if new_un_op > len(self.nop.get_unary_dict_keys()) - 1: return None
            new_cell = (node, new_un_op)
            _ = loc_psi[sublist_index].pop()
            loc_psi[sublist_index].append(new_cell)
            loc_psi[sublist_index].append(_)
        elif variation[0] == 3: # delete unary
            node_to_del = variation[2]
            if len(loc_psi[sublist_index]) < 3: return None
            elif node_to_del >= len(loc_psi[sublist_index]) - 1: return None
            else:
                for ind, sublist in enumerate(loc_psi[:sublist_index]):
                    if sublist[-1][0] == node_to_del:
                        nodes = [list(map(itemgetter(0), sublist[:-1])) for sublist in loc_psi[ind + 1:]]
                        break
                else:
                    nodes = [list(map(itemgetter(0), sublist[:-1])) for sublist in loc_psi]
                if sum(x.count(node_to_del) for x in nodes) > 1:
                    del loc_psi[sublist_index][node_to_del]
                else:
                    return None
        elif variation[0] == 4: # change parameter
            param_index = variation[1]
            new_value = variation[2]
            self.nop.variate_parameters(param_index, new_value)
                
    def cross(self, population, ksearch, var_num, children_num=8):
        best_individual, best_value = self.get_best_individual(population)
        parent1, parent1_est = self.get_best_individual(population, ksearch=ksearch)
        parent2, parent2_est = self.get_best_individual(population, ksearch=ksearch)
        if np.max([best_value/parent1_est, best_value/parent2_est]) > np.random.uniform():
            param_len = len(self.nop.get_q())
            all_variations = np.vstack((parent1, parent2))
            new_vars_len = round(0.38 * all_variations.shape[0])
            _ = [self.generate_variation(mutate_param=i%param_len) for i in range(new_vars_len)]
            _ = np.reshape(_, (-1, 4))
            all_variations = np.vstack((all_variations, _))
            sex = lambda: all_variations[np.random.choice(all_variations.shape[0], var_num, replace=False), :]
            ch = [sex() for i in range(children_num)]
            children = {}
            for child in ch:
                functional = self.estimate_variation(child)
                children[functional] = child
            return children
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
    
    def emigrate(self, pop_len_before, variations_per_individual, population):
        pop_len_after = len(population)
        emigration_len = pop_len_after - pop_len_before
        emigration = {}
        while len(population) < pop_len_before:
            individual = [self.generate_variation(mutate_param=False) for i in range(variations_per_individual)]
            functional = self.estimate_variation(individual)
            if functional < 1e+3: population[functional] = individual
        return population
    
    def change_base_psi(self, individual):
        """ Apply variations in individual to the structure
        :param individual: list of lists of objects
        :return: None
        """
        for variation in individual:
            self.apply_variation(variation)
        self.nop.update_base_psi()
        self.nop.update_base_q()

    def optimize(self, qmin=-10, qmax=10, individuals=100, generations=256, psi_change_epoch=2,
                       variations_per_individual=4, crossings=256, ksearch=16):
        self.qmin = qmin
        self.qmax = qmax
        parameters_len = len(self.nop.get_q())
        print('Initializing hyper-parameters...')
        print('''qmin: {0}, qmax: {1}\nPopulation size: {2}\nMax variations number for each individual: {3}\nGenerations: {4}\nCrossings per epoch: {5}\nksearch: {6}\nGeneration to change psi: each {7}th'''.format(
                 qmin, qmax, individuals, variations_per_individual, generations, crossings, ksearch, psi_change_epoch))
        print('Beginning structure synthesis with parameters: {0}'.format(list(self.nop.get_q().values())))
        print('Generating population for structure synthesis...')
        pop_gen_func = lambda: self.generate_variations_population(individuals, variations_per_individual)
        population = pop_gen_func()
        self.base_estimation = self.estimate_object_function() # estimation of base psi
        print('J: {0}'.format(self.base_estimation))
        for g in range(generations):
            print('Generation {0} is running...'.format(g))
            if g % psi_change_epoch == 0 and g != 0 and generations > psi_change_epoch:
                candidate = self.estimate_variation(best_individual)
                if candidate < self.base_estimation:
                    self.change_base_psi(best_individual)
                    self.base_estimation = candidate
                else:
                    print('Proceeding with no changes applied to structure and parameters')
                    continue
                print('Refreshing parameters: \n{0}'.format(list(self.nop.get_q().values())))
                print('Refreshing base structure: \n{0}'.format(self.nop.get_base_psi()))
                print('J: {0}'.format(best_value))
                print("#####################################################")
            for c in range(crossings):
                children = self.cross(population, ksearch, variations_per_individual, children_num=4)
                if children is not None:
                    population = self.insert_children(population, children)
                    population = self.emigrate(individuals, variations_per_individual, population)
                else: continue
            best_individual, best_value = self.get_best_individual(population)
        candidate = self.estimate_variation(best_individual)
        if candidate < self.base_estimation:
            self.change_base_psi(best_individual) # final psi/q change
            self.base_estimation = candidate
        print('Optimizing parameters with fixed structure:\n{0}'.format(self.nop.get_base_psi()))
        gaussian_means = list(self.nop.get_base_q().values())
        pg = BaseGenetics(e=gaussian_means)
        pg.set_estimator(self.estimate_parameters)
        best_q, best_estimation = pg.optimize(qmin, qmax,
                                              individuals,
                                              generations,
                                              parameters_len,
                                              crossings,
                                              ksearch)
        print('Done.')
        print('J: {0}\nQ: {1}\nStructure:\n{2}'.format(best_estimation, best_q, self.nop.get_base_psi()))
        

