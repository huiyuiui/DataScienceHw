# you must use python 3.10
# For linux, you must use download HomeworkFramework.cpython-310-x86_64-linux-gnu.so
# For Mac, you must use download HomeworkFramework.cpython-310-darwin.so
# If above can not work, you can use colab and download HomeworkFramework.cpython-310-x86_64-linux-gnu.so and don't forget to modify output's name.

import numpy as np
from HomeworkFramework import Function # type: ignore
import cma


class Optimizer(Function): # need to inherit this class "Function"
    def __init__(self, target_func):
        super().__init__(target_func) # must have this init to work normally

        self.lower = self.f.lower(target_func)
        self.upper = self.f.upper(target_func)
        self.dim = self.f.dimension(target_func)

        self.target_func = target_func

        self.eval_times = 0
        self.optimal_value = float("inf")
        self.optimal_solution = np.empty(self.dim)

        sigma = 0.3 * (self.upper - self.lower)
        self.cma = cma.CMAEvolutionStrategy(self.dim * [0.5 * (self.upper + self.lower)], sigma, {'bounds': [self.lower, self.upper], 'maxfevals': 5000})


    # def get_optimal(self):
    #     return self.optimal_solution, self.optimal_value

    # def run(self, FES): # main part for your implementation
        
    #     while self.eval_times < FES:
    #         print('=====================FE=====================')
    #         print(self.eval_times)

    #         solution = np.random.uniform(np.full(self.dim, self.lower), np.full(self.dim, self.upper), self.dim)
    #         value = self.f.evaluate(func_num, solution)
    #         self.eval_times += 1

    #         if value == "ReachFunctionLimit":
    #             print("ReachFunctionLimit")
    #             print(self.eval_times)
    #             break            
    #         if float(value) < self.optimal_value:
    #             self.optimal_solution[:] = solution
    #             self.optimal_value = float(value)

    #         print("optimal: {}\n".format(self.get_optimal()[1]))
    
    def get_optimal(self):
        return self.cma.result.xbest, self.cma.result.fbest

    def run(self, fes):
        while not self.cma.stop():

            solutions = self.cma.ask()
            values = []

            for x in solutions:
                value = self.f.evaluate(self.target_func, x)
                self.eval_times += 1

                if self.eval_times >= fes:
                    print("Evaluate times:", self.eval_times)
                    return

                if value == "ReachFunctionLimit":
                    print("ReachFunctionLimit")
                    return
                
                values.append(value)

            self.cma.tell(solutions, values)
            self.cma.disp()

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

        # you should implement your optimizer
        op = Optimizer(func_num)
        op.run(fes)
        
        best_input, best_value = op.get_optimal()
        print(best_input, best_value)
        
        # change the name of this file to your student_ID and it will output properlly
        with open("{}_function{}.txt".format(__file__.split('_')[0], func_num), 'w+') as f:
            for i in range(op.dim):
                f.write("{}\n".format(best_input[i]))
            f.write("{}\n".format(best_value))

        func_num += 1 
        print()