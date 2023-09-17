class BayesianOptimization:
    def __init__(self, model, param_space):
        """
        Initialize the Bayesian Optimization class.
        
        Parameters:
        model: The deep learning model to be optimized.
        param_space: A dictionary defining the hyperparameters search space.
        """
        self.model = model
        self.param_space = param_space

    def objective_function(self, params):
        """
        Define the objective function to be optimized. This function should take a set of hyperparameters and return the evaluation metric for those hyperparameters.
        
        Parameters:
        params: A set of hyperparameters from the search space.
        """
        pass

    def optimize(self, n_iter):
        """
        Perform the Bayesian optimization over the hyperparameters space.
        
        Parameters:
        n_iter: The number of iterations to perform.
        """
        pass

    def get_best_params(self):
        """
        Return the best set of hyperparameters found during the optimization.
        """
        pass
