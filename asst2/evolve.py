import numpy as np
from random import random, sample
def choose_parents(pop, k):
    inds = sample(range(len(pop)), k)
    b_i = None
    b2_i = None
    b = 0
    b2 = 0
    small_pop = []
    for i in inds:
        if pop[i][1] > b:
            b = pop[i][1]
            b_i = i
        elif pop[i][1] > b2:
            b2 = pop[i][1]
            b2_i = i
    return b_i, b2_i


class GeneticAlgorithm:
    def __init__(self,  crossover='uniform', num_steps=5000, pop_size=25, mutation_rate=0.05, ONE_RATE = 0.5):
        self.crossover = crossover
        self.num_steps = num_steps
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.ONE_RATE = ONE_RATE
    
    def evolve(self, x, y, eval_func, classifier, verbose=False):
        population = []
        best_mask = None
        best_accuracy = 0
        for _ in range(self.pop_size):
            mask = []
            for u2 in range(len(x[0])):
                if random() < self.ONE_RATE:
                    mask.append(1)
                else:
                    mask.append(0)
            population.append(mask)
        with open("rbsvm_log.txt", "w") as f:
            pass
        for i, p in enumerate(population):
            p_x = []
            for x_i in x:
                p_x.append(np.array([e for pos, e in enumerate(x_i) if p[pos]]))
            p_x = np.array(p_x)
            acc = eval_func(classifier, list(p_x), list(y))
            population[i] = [p, acc]
        for step in range(self.num_steps):
            mom_score = 0 # best score
            dad_score = 0 # second best score
            worst = float('inf')
            # mom_position = len(population) - 1
            # dad_position = len(population) - 2
            mom_position, dad_position = choose_parents(population, 8)
            # for i, p in enumerate(population):
                # p_x = []
                # for x_i in x:
                #     p_x.append(np.array([e for pos, e in enumerate(x_i) if p[pos]]))
                # p_x = np.array(p_x)
                # acc = eval_func(classifier, p_x, y)
                # if acc > mom_score:
                #     mom_score = acc
                #     mom_position = i
                # elif acc > dad_score and acc < mom_score:
                #     dad_score = acc
                #     dad_position = i
                # if acc < worst:
                #     worst = acc
                #     worst_position = i
            population.sort(key=lambda x: x[1])
            
            worst = population[0][1]
            mom_score = population[mom_position][1]
            dad_score = population[dad_position][1]

            # if mom_score > best_accuracy:
            #     best_accuracy = mom_score
            #     best_mask = population[mom_position]
            best_accuracy = population[len(population) - 1][1]
            best_mask = population[len(population) - 1][0]

            baby = []
            mom_arr = population[mom_position][0]
            dad_arr = population[dad_position][0]
            for bit in range(len(mom_arr)):
                if random() < 0.5:
                    baby.append(mom_arr[bit])
                else:
                    baby.append(dad_arr[bit])
            for i, bit in enumerate(baby):
                if random() < self.mutation_rate:
                    if bit:
                        baby[i] = 0
                    else:
                        baby[i] = 1
            baby_p_x = []
            for x_i in x:
                baby_p_x.append(np.array([e for pos, e in enumerate(x_i) if baby[pos]]))
            baby_p_x = np.array(baby_p_x)
            baby_acc = eval_func(classifier, baby_p_x, y)
            population[0] = [baby, baby_acc]        
            with open("steadystate_log.txt", "a") as f:
                f.write(",".join([str(step), str(mom_score),str(worst)]) + "\n")
            if step % 20 == 0:
                print("step " + str(step) + " / " + str(self.num_steps))
                print("best_acc", best_accuracy)
                print("mom", mom_score, mom_position)
                print("dad", dad_score, dad_position)
                print("worst", worst, 0)
                print("best mask", best_mask)
                print()


                

