# you must use python 3.10
# For linux, you must use download HomeworkFramework.cpython-310-x86_64-linux-gnu.so
# For Mac, you must use download HomeworkFramework.cpython-310-darwin.so
# If above can not work, you can use colab and download HomeworkFramework.cpython-310-x86_64-linux-gnu.so and don't forget to modify output's name.

import numpy as np
from HomeworkFramework import Function # type: ignore
import cma

class CMA_ES_Optimizer(Function): # need to inherit this class "Function"
    def __init__(self, target_func):
        super().__init__(target_func)  # must have this init to work normally

        # function parameters
        self.lower = self.f.lower(target_func)
        self.upper = self.f.upper(target_func)
        self.dim = self.f.dimension(target_func)

        self.target_func = target_func

        self.eval_times = 0
        self.optimal_value = float("inf")
        self.optimal_solution = np.empty(self.dim)

        # CMA-ES parameters
        self.expected = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * np.square(self.dim)))
        self.step_size = 0.3 * np.max(self.upper - self.lower)

        self.offspring_size = int(4 + np.floor(3 * np.log(self.dim)))
        self.parent_size = int(np.floor(self.offspring_size / 2))

        self.weights = np.zeros(self.parent_size, dtype=float)
        for i in range(1, self.parent_size + 1):
            self.weights[i - 1] = np.log((self.offspring_size + 1) / 2) - np.log(i)
        self.weights = self.weights / np.sum(self.weights)
        self.mu_w = 1 / np.sum(np.square(self.weights))

        self.c_sigma = (self.mu_w + 2) / (self.dim + self.mu_w + 5)
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mu_w - 1) / (self.dim + 1)) - 1) + self.c_sigma

        self.c_c = (4 + self.mu_w / self.dim) / (self.dim + 4 + 2 * self.mu_w / self.dim)
        self.c_1 = 2 / (np.square(self.dim + 1.3) + self.mu_w)
        self.c_mu = min(1 - self.c_1, 2 * ((self.mu_w - 2 + 1 / self.mu_w) / (np.square(self.dim + 2) + 2 * self.mu_w / 2)))

        # matrix init
        self.b_mat = np.identity(self.dim)
        self.diagonal = np.ones(self.dim)
        self.cov_mat = np.identity(self.dim)
        self.inv_cov_mat = np.identity(self.dim)

        self.offsprings = np.full(self.offspring_size, None)
        self.parents = np.full(self.parent_size, None)

        self.mean = np.random.uniform(self.lower, self.upper, self.dim)
        self.prev_mean = np.copy(self.mean)

        self.p_c = np.zeros(self.dim)
        self.p_sigma = np.zeros(self.dim)
        self.y_w = np.zeros(self.dim)

    def sample_offspring(self):
        for i in range(self.offspring_size):
            self.offsprings[i] = self.mean + self.step_size * np.dot(self.b_mat, self.diagonal * np.random.randn(self.dim))
            np.clip(self.offsprings[i], self.lower, self.upper, out=self.offsprings[i])

    def select_offspring(self):
        values =[]
        for i in range(self.offspring_size):
            value = self.f.evaluate(self.target_func, self.offsprings[i])
            self.eval_times += 1
            if value == "ReachFunctionLimit":
                # print("ReachFunctionLimit")
                # print("Evaluation times:", self.eval_times)
                return 1
            values.append(value)

        selected_index = np.argsort(values)[:self.parent_size]
        self.parents = self.offsprings[selected_index]
        if(values[selected_index[0]] < self.optimal_value):
            self.optimal_value = values[selected_index[0]]
            self.optimal_solution = self.parents[0]
        
        return 0

    def update_mean(self):
        self.prev_mean = self.mean
        now_mean = np.zeros(self.dim)
        for i in range(self.parent_size):
            now_mean += self.weights[i] * self.parents[i]
        self.mean = now_mean
        self.y_w = (self.mean - self.prev_mean) / self.step_size

    def update_cov_mat(self):
        self.p_c = (1 - self.c_c) * self.p_c + np.sqrt(self.c_c * (2 - self.c_c) * self.mu_w) * self.y_w
        rank_one_update = self.c_1 * np.outer(self.p_c, self.p_c)
        rank_mu_update = np.zeros((self.dim, self.dim))
        for i in range(self.parent_size):
            y = (self.parents[i] - self.prev_mean) / self.step_size
            rank_mu_update += self.weights[i] * np.outer(y, y)
        rank_mu_update *= self.c_mu
        self.cov_mat = (1 - self.c_1 - self.c_mu) * self.cov_mat + rank_one_update + rank_mu_update

    def update_step_size(self):
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_w) * np.dot(self.inv_cov_mat, self.y_w)
        self.step_size = self.step_size * np.exp(self.c_sigma / self.d_sigma * (np.linalg.norm(self.p_sigma) / self.expected - 1))

    def cov_mat_decompose(self):
        self.cov_mat = (self.cov_mat + self.cov_mat.T) / 2
        eigenvalues, eigenvectors = np.linalg.eigh(self.cov_mat)
        self.b_mat = eigenvectors
        self.diagonal = np.sqrt(eigenvalues)
        inv_sqrt_diag = np.diag(1.0 / self.diagonal)
        self.inv_cov_mat = np.dot(np.dot(self.b_mat, inv_sqrt_diag), self.b_mat.T)

    def get_optimal(self):
        return self.optimal_solution, self.optimal_value

    def run(self, FES): # main part for your implementation

        while self.eval_times < FES:
            # print('=====================FE=====================')
            # print(self.eval_times)
            
            self.sample_offspring()
            is_done = self.select_offspring()
            if is_done == 1:
                break
            self.update_mean()
            self.update_cov_mat()
            self.update_step_size()
            self.cov_mat_decompose()

            # print("optimal: {}\n".format(self.get_optimal()[1]))

if __name__ == '__main__':
    func_num = 1
    fes = 0
    #function1: 1000, function2: 1500, function3: 2000, function4: 2500
    while func_num < 5:
        if func_num == 1:
            fes = 1000
        elif func_num == 2:
            fes = 1500
        elif func_num == 3:
            fes = 2000 
        else:
            fes = 2500

        best_input = []
        best_value = float("inf")

        # avoid randomness
        for i in range(10):
            # you should implement your optimizer
            op = CMA_ES_Optimizer(func_num)
            op.run(fes)
            
            opt_input, opt_value = op.get_optimal()
            if(opt_value < best_value):
                best_input = opt_input
                best_value = opt_value       

        print("Best input:", best_input)
        print("Best value:", best_value)     
        
        # change the name of this file to your student_ID and it will output properlly
        with open("{}_function{}.txt".format(__file__.split('_')[0], func_num), 'w+') as f:
            for i in range(op.dim):
                f.write("{}\n".format(best_input[i]))
            f.write("{}\n".format(best_value))

        func_num += 1 
        print()