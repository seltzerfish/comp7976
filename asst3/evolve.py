import numpy as np
from random import random, sample, randint
def choose_parents_tournament(pop, k):
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

def choose_losers_tournament(pop, k):
    inds = sample(range(len(pop)), k)
    b_i = None
    b2_i = None
    b = float("inf")
    b2 = float("inf")
    small_pop = []
    for i in inds:
        if pop[i][1] < b:
            b = pop[i][1]
            b_i = i
        elif pop[i][1] < b2:
            b2 = pop[i][1]
            b2_i = i
    return b_i, b2_i


class SteadyStateGA:
    def __init__(self,  crossover='uniform', num_steps=5000, pop_size=25, mutation_rate=0.05, ONE_RATE = 0.5):
        self.crossover = crossover
        self.num_steps = num_steps
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.ONE_RATE = ONE_RATE
    
    def evolve(self, x, y, eval_func, classifier, trial, verbose=False):
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
        with open("logs/steady/" + str(trial) + "steady.txt", "w") as f:
            f.write("\n")
        for i, p in enumerate(population):
            p_x = []
            for x_i in x:
                p_x.append(np.array([e for pos, e in enumerate(x_i) if p[pos]]))
            p_x = np.array(p_x)
            acc = eval_func(classifier, p_x, y)
            population[i] = [p, acc]
        for step in range(self.num_steps):
            mom_score = 0 # best score
            dad_score = 0 # second best score
            worst = float('inf')
            population.sort(key=lambda x: x[1])
            mom_position, dad_position = choose_parents_tournament(population, 8)
            
            worst = population[0][1]
            mom_score = population[mom_position][1]
            dad_score = population[dad_position][1]

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
            with open("logs/steady/" + str(trial) + "steady.txt", "a") as f:
                f.write(",".join([str(step), str(best_accuracy),str(worst)]) + "\n")
            if step % 20 == 0:
                print("step " + str(step) + " / " + str(self.num_steps))
                print("best_acc", best_accuracy)
                print("mom", mom_score, mom_position)
                print("dad", dad_score, dad_position)
                print("worst", worst, 0)
                print("best mask", best_mask)
                print()
        with open("best_masks_steady.txt", "a") as f:
            f.write(str(best_mask) + " - " + str(best_accuracy) + "\n")
        return best_mask

class ElitistGA:
    def __init__(self,  crossover='uniform', num_steps=5000, pop_size=25, mutation_rate=0.05, ONE_RATE = 0.5):
        self.crossover = crossover
        self.num_steps = num_steps
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.ONE_RATE = ONE_RATE
    
    def evolve(self, x, y, eval_func, classifier, trial, verbose=False):
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
        with open("logs/elite/" + str(trial) + "elite.txt", "w") as f:
            f.write("\n")
        for i, p in enumerate(population):
            p_x = []
            for x_i in x:
                p_x.append(np.array([e for pos, e in enumerate(x_i) if p[pos]]))
            p_x = np.array(p_x)
            acc = eval_func(classifier, p_x, y)
            population[i] = [p, acc]
        for step in range(self.num_steps):
            mom_score = 0 # best score
            dad_score = 0 # second best score
            worst = float('inf')
            population.sort(key=lambda x: x[1])
            mom_position, dad_position = choose_parents_tournament(population, 32)
            
            worst = population[0][1]
            mom_score = population[mom_position][1]
            dad_score = population[dad_position][1]

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
            if baby_acc > worst:
                population[0] = [baby, baby_acc]        
            with open("logs/elite/" + str(trial) + "elite.txt", "a") as f:
                f.write(",".join([str(step), str(best_accuracy),str(worst)]) + "\n")
            if step % 20 == 0:
                print("step " + str(step) + " / " + str(self.num_steps))
                print("best_acc", best_accuracy)
                print("mom", mom_score, mom_position)
                print("dad", dad_score, dad_position)
                print("worst", worst, 0)
                print("best mask", best_mask)
                print()
        with open("best_masks_elitist.txt", "a") as f:
            f.write(str(best_mask) + " - " + str(best_accuracy) + "\n")
        return best_mask


class SteadyStateGA:
    def __init__(self,  crossover='uniform', num_steps=5000, pop_size=25, mutation_rate=0.05, ONE_RATE = 0.5):
        self.crossover = crossover
        self.num_steps = num_steps
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.ONE_RATE = ONE_RATE
    
    def evolve(self, x, y, eval_func, classifier, trial, verbose=False):
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
        with open("logs/steady/" + str(trial) + "steady.txt", "w") as f:
            f.write("\n")
        for i, p in enumerate(population):
            p_x = []
            for x_i in x:
                p_x.append(np.array([e for pos, e in enumerate(x_i) if p[pos]]))
            p_x = np.array(p_x)
            acc = eval_func(classifier, p_x, y)
            population[i] = [p, acc]
        for step in range(self.num_steps):
            mom_score = 0 # best score
            dad_score = 0 # second best score
            worst = float('inf')
            population.sort(key=lambda x: x[1])
            mom_position, dad_position = choose_parents_tournament(population, 8)
            
            worst = population[0][1]
            mom_score = population[mom_position][1]
            dad_score = population[dad_position][1]

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
            with open("logs/steady/" + str(trial) + "steady.txt", "a") as f:
                f.write(",".join([str(step), str(best_accuracy),str(worst)]) + "\n")
            if step % 20 == 0:
                print("step " + str(step) + " / " + str(self.num_steps))
                print("best_acc", best_accuracy)
                print("mom", mom_score, mom_position)
                print("dad", dad_score, dad_position)
                print("worst", worst, 0)
                print("best mask", best_mask)
                print()
        with open("best_masks_steady.txt", "a") as f:
            f.write(str(best_mask) + " - " + str(best_accuracy) + "\n")
        return best_mask

class EstimationGA:
    def __init__(self,  crossover='uniform', num_steps=5000, pop_size=25, mutation_rate=0.05, ONE_RATE = 0.5):
        self.crossover = crossover
        self.num_steps = num_steps
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.ONE_RATE = ONE_RATE
    
    def evolve(self, x, y, eval_func, classifier, trial, mu=8, verbose=False):
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
        with open("logs/eda/" + str(trial) + "eda.txt", "w") as f:
            f.write("\n")
        for i, p in enumerate(population):
            p_x = []
            for x_i in x:
                p_x.append(np.array([e for pos, e in enumerate(x_i) if p[pos]]))
            p_x = np.array(p_x)
            acc = eval_func(classifier, p_x, y)
            population[i] = [p, acc]
        for step in range(self.num_steps):
            mom_score = 0 # best score
            dad_score = 0 # second best score
            worst = float('inf')
            inds = sample(range(len(population)), mu)
            sub_pop_parents = []
            for ind in inds:
                sub_pop_parents.append(population[ind][0])
            sub_pop_children = []
            for _2 in range(len(population)):
                sub_pop_child = []
                for o in range(len(sub_pop_parents[0])):
                    sub_pop_child.append(sub_pop_parents[randint(0, mu - 1)][o])
                sub_pop_children.append(sub_pop_child)
            for o, kid in enumerate(sub_pop_children):
                kid_p_x = []
                for x_i in x:
                    kid_p_x.append(np.array([e for pos, e in enumerate(x_i) if kid[pos]]))
                kid_p_x = np.array(kid_p_x)
                kid_acc = eval_func(classifier, kid_p_x, y)
                sub_pop_children[o] = (kid, kid_acc)
            sub_pop_children.sort(key=lambda x: x[1])
            for ind in inds:
                population[ind] = sub_pop_children.pop()
            population.sort(key=lambda x: x[1])
            worst = population[0][1]
            if population[len(population) - 1][1] > best_accuracy:
                best_accuracy = population[len(population) - 1][1]
                best_mask = population[len(population) - 1][0]
            # mom_position, dad_position = choose_parents_tournament(population, 8)
            
            # worst = population[0][1]
            # mom_score = population[mom_position][1]
            # dad_score = population[dad_position][1]

            # best_accuracy = population[len(population) - 1][1]
            # best_mask = population[len(population) - 1][0]

            # baby = []
            # mom_arr = population[mom_position][0]
            # dad_arr = population[dad_position][0]
            # for bit in range(len(mom_arr)):
            #     if random() < 0.5:
            #         baby.append(mom_arr[bit])
            #     else:
            #         baby.append(dad_arr[bit])
            # for i, bit in enumerate(baby):
            #     if random() < self.mutation_rate:
            #         if bit:
            #             baby[i] = 0
            #         else:
            #             baby[i] = 1
            # baby_p_x = []
            # for x_i in x:
            #     baby_p_x.append(np.array([e for pos, e in enumerate(x_i) if baby[pos]]))
            # baby_p_x = np.array(baby_p_x)
            # baby_acc = eval_func(classifier, baby_p_x, y)
            # if baby_acc > worst:
            #     population[0] = [baby, baby_acc]        
            with open("logs/eda/" + str(trial) + "eda.txt", "a") as f:
                f.write(",".join([str(step), str(best_accuracy),str(worst)]) + "\n")
            
            print("step " + str(step) + " / " + str(self.num_steps))
            print("best_acc", best_accuracy)
            # print("mom", mom_score, mom_position)
            # print("dad", dad_score, dad_position)
            print("worst", worst, 0)
            print("best mask", best_mask)
            print()
        with open("best_masks_eda.txt", "a") as f:
            f.write(str(best_mask) + " - " + str(best_accuracy) + "\n")
        return best_mask