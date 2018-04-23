from copy import deepcopy, copy

class NetworkOperator(object):
    """
    
    """
    def __init__(self, unaries, binaries, input_nodes, output_nodes):
        """
        Instantiates Network Operator object
        xq - list of variables and parameters
        unaries - list of unary functions
        binaries - list of binary functions
        """
        self.psi = None
        self.base_psi = None
        self.un_dict = {ind: func for ind, func in enumerate(unaries)}
        self.bin_dict = {ind: func for ind, func in enumerate(binaries)}
        self.q = []
        self.base_q = []
        self.__input_node_free_index = None
        self.output_nodes = output_nodes # list of indexes that output nodes posess
        self.input_nodes = input_nodes # list of indexes that input nodes posess
  
    def get_input_nodes(self,):
        return self.input_nodes
    
    def set_q(self, q):
        """
        q - list
        return dict
        """
        self.q = {ind: val for ind, val in enumerate(q)}
        self.__input_node_free_index = len(q)

    def get_q(self,):
        return self.q
    
    def set_base_q(self, q):
        self.base_q = {ind: val for ind, val in enumerate(q)}
        self.__input_node_free_index = len(q)
        
    def get_base_q(self,):
        return self.base_q
    
    def update_base_q(self,):
        new_q = copy(self.get_q()).values()
        self.set_base_q(new_q)
        
    def roll_back_to_base_q(self,):
        old_q = copy(self.get_base_q()).values()
        self.set_q(old_q)
        
    def get_free_input_node(self,):
        return self.__input_node_free_index
    
    def variate_parameters(self, index, value):
        q = self.get_q()
        q[index] = value
        
    def get_psi(self,):
        return self.psi

    def set_psi(self, psi):
        self.psi = psi 
    
    def get_base_psi(self,):
        return self.base_psi
        
    def set_base_psi(self, base_psi):
        self.base_psi = base_psi
        
    def update_base_psi(self,):
        new_psi = deepcopy(self.get_psi())
        self.set_base_psi(new_psi)
        
    def roll_back_to_base_psi(self,):
        old_psi = deepcopy(self.get_base_psi())
        self.set_psi(old_psi)
    
    def get_unary_dict_keys(self,):
        return list(self.un_dict.keys())
    
    def get_binary_dict_keys(self,):
        return list(self.bin_dict.keys())
    
    def eval_psi(self, x):
        """
        out_nodes - indexes of nodes which are outputs of nop. [list]
        x - list of state components
        """
        x = {self.get_free_input_node() + ind: val for ind, val in enumerate(x)}
        xq = {**self.q, **x} # merge two dicts without altering the originals
        d = {} # node: value
        psi = self.get_psi()
        def apply_unary(unary_func, unary_index):
            try: return unary_func(xq[unary_index]) # try to apply unary to q or x
            except: return unary_func(d[unary_index]) # apply unary to a dictinoary with that node otherwise
        for cell in psi:
            binary_index = cell[-1][1] # binary operation index
            binary_func = self.bin_dict[binary_index] # binary function object
            d[cell[-1][0]] = binary_func([apply_unary(self.un_dict[i[1]], i[0]) for i in cell[:-1]])
        nop_outputs = tuple([d[node] for node in self.output_nodes])
        return nop_outputs

