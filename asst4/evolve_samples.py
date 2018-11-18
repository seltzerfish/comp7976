import numpy as np
from random import random, sample, randint
from pprint import pprint
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

def make_babies_rc(mom, dad, std, mutation_rate):
    babies = []
    babies.append(0.5*mom + 0.5*dad)
    babies.append(1.5*mom - 0.5*dad)
    babies.append(1.5*dad - 0.5*mom)
    babies.append(1.5*mom + 1.5*dad)
    for j in range(len(babies)):
        newb = []
        for i, e in enumerate(babies[j]):
            if random() < mutation_rate:
                if random() < 0.5:
                    newb.append(e + random() * std[i])
                else:
                    newb.append(e + -1*(random() * std[i]))
                    pass
            else:
                newb.append(e)
        babies[j] = np.array(newb)

    return babies

def make_babies_eda(population, mu, mutation_rate):
    inds = sample(range(len(population)), mu)
    sub_pop_parents = []
    for ind in inds:
        sub_pop_parents.append(population[ind][0])
    sub_pop_children = []
    mean = np.mean(sub_pop_parents, axis=0)
    std = np.std(sub_pop_parents, axis=0)
    for _ in range(len(population)):
        sub_pop_children.append(np.random.normal(mean, std))
    tot_std = np.std(np.array([p[0] for p in population]), axis=0)
    for child in sub_pop_children:
        for i in range(len(child)):
            if random()< mutation_rate:
                if random() < 0.5:
                    child[i] += random() * tot_std[i]
                else:
                    child[i] -= random() * tot_std[i]

    return sub_pop_children, inds


def choose_parents_tournament_lowest(pop, k):
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
    def __init__(self,  crossover='uniform', num_steps=700000, pop_size=25, mutation_rate=0.05, ONE_RATE = 0.5):
        self.crossover = crossover
        self.num_steps = num_steps
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.ONE_RATE = ONE_RATE
    
    def evolve(self, x, mask, eval_func, classifier,goal, trial, verbose=False):
        population = []
        best_vector = None
        lowest_diff = float("inf")
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        for _ in range(self.pop_size):
            population.append(np.random.normal(mean, std))
        with open("logs/steady/" + str(trial) + "steady.txt", "w") as f:
            f.write("\n")
        for i, p in enumerate(population):
            newp = []
            for j, e in enumerate(p):
                if mask[j]:
                    newp.append(e)
            acc = eval_func(classifier, goal, np.array(newp))
            # print(newp)
            population[i] = [np.array(newp), acc]
        # pprint(population)
        for step in range(self.num_steps):
            population.sort(key=lambda x: x[1], reverse=True)
            mom_position, dad_position = choose_parents_tournament_lowest(population, 34)
            # print(population)
            # exit()
            worst = population[0][1]
            mom_score = population[mom_position][1]
            dad_score = population[dad_position][1]

            lowest_diff = population[len(population) - 1][1]
            best_vector = population[len(population) - 1][0]

            mom_arr = population[mom_position][0]
            dad_arr = population[dad_position][0]
            babies = make_babies_rc(mom_arr, dad_arr, std, self.mutation_rate)
            best_baby_acc = float("inf")
            best_baby = None
            for baby in babies:
                b_acc = eval_func(classifier, goal, baby)
                if b_acc < best_baby_acc:
                    best_baby = baby
                    best_baby_acc = b_acc
        
            population[0] = [best_baby, best_baby_acc]        
            with open("logs/steady/" + str(trial) + "steady.txt", "a") as f:
                f.write(",".join([str(step), str(lowest_diff),str(worst)]) + "\n")
            if step % 100 == 0:
                print("step " + str(step) + " / " + str(self.num_steps))
                print("lowest diff", lowest_diff)
                print("mom", mom_score, mom_position)
                print("dad", dad_score, dad_position)
                print("worst", worst)
                print("best vector", best_vector)
                letters = ['#', '%', "'", '(', ')', '*', '-', '.', '3', '7', '8', '9', ';', '>', '?', 'A', 'B', 'C', 'E', 'F', 'K', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'W', '[', '\\', ']', 'a', 'b', 'c', 'd', 'i', 'j', 'k', 'm', 'n', 'o', 'p', 'r', 's', 'u', 'v', 'w', 'y']
                # bv = [[letters[i], e] for i, e in enumerate(best_vector)]
                # print("best vector", bv)
                print()
        with open("best_masks_steady.txt", "a") as f:
            f.write(str(best_vector) + " - " + str(lowest_diff) + "\n")
        return best_vector

class ElitistGA:
    def __init__(self,  crossover='uniform', num_steps=50000, pop_size=100, mutation_rate=0.05, ONE_RATE = 0.5):
        self.crossover = crossover
        self.num_steps = num_steps
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.ONE_RATE = ONE_RATE
    
    def evolve(self, x, mask, eval_func, classifier,goal, trial, verbose=False):
        population = []
        best_vector = None
        lowest_diff = float("inf")
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        for _ in range(self.pop_size):
            population.append(np.random.normal(mean, std))
        population[0] = [373323.452, 1407719.8, 188976.462, -1489849.13, 577310.281, 520470.6, -575981.511, -183290.28, -70321.5667, 927997.78, -4787821.01, 829682.74, 2117900.17, 3947449.57, 6442849.7, -2125084.06, -1480070.97, -3468838.84, 819169.467, 31741.9946, -120145.625, -990352.031, -6303261.86, 322839.192, -6371960.05, -3379714.62, -228137.163, 1220173.13, -9092947.39, 1469982.83, -1468040.45, 323107.506, -2463974.11, 122177.587, 1814792.32, -134373.965, 1761792.32, -75691.2489, 131350.479, 343423.827, 1335695.16, -63872.7987, 6311.15526, -39642.5759, 275346.787, 628573.57, 92782.617, 183829.871, 175286.039, -1024878.49, 1191063.47, 688771.275, 50549.7736, 615061.857, -756147.671, 315625.049, -7330.20751, -606161.151, 207899.403, 2379.29348, -104719.734, 0.0, 95960.3145, -644923.668, 0.0, 0.0, 41983.976, 456073.497, 379574.087, -137132.831, 763585.585, 0.0, 19022.4963, 2961174.17, -212151.912, -1652401.38, 92166.0489, -309076.266, 23406.6107, 0.0, 0.0,0.0, -84231.1863, 143888.971, 5936.0006, 0.0, -9642.64106, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -6566962.71, 0.0, -1589688.57, 0.0, 0.0, 0.0]
        population[1] = [0.0102769128, 0.00959128401, 0.00559280225, 0.010926068, 0.00865290435, 0.00773617994, 0.00363192238, 0.00709649304, 0.00332049016, 0.00509915694, 0.0859125344, 0.0526173603, 0.0825408709, 0.103107078, 0.37933068, -211.07779, 0.0955826445, 138.754103, 0.241091416, 0.0015244321, 0.0442069593, 0.112583391, 0.0651905684, 0.233118289, 0.211492221, 0.0598735053, 0.00230510211, 227.485445, -28.2724963, 0.213842244, 0.0990022789, 0.0322620623, 0.0328883259, 0.00855515291, 0.0677074676, 0.0038630006, 0.0165052917, 0.00365778738, 0.0115829832, 0.0135738985, 0.00436094375, 0.00342072619, 0.00286586204, 0.0100823665, 0.0197426248, 0.00541267565, 0.00151041272, 0.00144073209, 0.00936369539, 0.0059014879, 0.00614298701, 0.000239854626, 0.000883049722, 0.00511949986, 0.01105606, 0.0125073462, 0.00116241826, 5.18901271e-07, 0.00238538343, -6.88921515e-05, 0.000308142286, 0.0, 0.000199774547, 0.00858015091, 0.0, 0.0, -0.00057694521, 0.000873434671, 0.0117745659, 0.00144544883, 0.00278023522, 0.0, 0.000312741729, 0.0367203638, 0.0129519313, 0.0306838168, -0.000258398053, 0.00495747412, -0.000177004489, 0.0, 0.0, 0.0, -0.000457863981, 0.000218524094, -4.92246385e-05, 0.0, -0.00012954812, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -160.565479, 0.0, -168.413115, 0.0, 0.0, 0.0]
        
        with open("logs/elite/" + str(trial) + "elite.txt", "w") as f:
            f.write("\n")
        for i, p in enumerate(population):
            newp = []
            for j, e in enumerate(p):
                if mask[j]:
                    newp.append(e)
            acc = eval_func(classifier, goal, np.array(newp))
            # print(newp)
            population[i] = [np.array(newp), acc]
        # pprint(population)
        for step in range(self.num_steps):
            population.sort(key=lambda x: x[1], reverse=True)
            mom_position, dad_position = choose_parents_tournament_lowest(population, 34)
            # print(population)
            # exit()
            worst = population[0][1]
            mom_score = population[mom_position][1]
            dad_score = population[dad_position][1]

            lowest_diff = population[len(population) - 1][1]
            best_vector = population[len(population) - 1][0]

            mom_arr = population[mom_position][0]
            dad_arr = population[dad_position][0]
            babies = make_babies_rc(mom_arr, dad_arr, std, self.mutation_rate)
            best_baby_acc = float("inf")
            best_baby = None
            for baby in babies:
                b_acc = eval_func(classifier, goal, baby)
                if b_acc < best_baby_acc:
                    best_baby = baby
                    best_baby_acc = b_acc
            if population[0][1] > best_baby_acc:
                population[0] = [best_baby, best_baby_acc]        
            with open("logs/elite/" + str(trial) + "elite.txt", "a") as f:
                f.write(",".join([str(step), str(lowest_diff),str(worst)]) + "\n")
            if step % 100 == 0:
                print("step " + str(step) + " / " + str(self.num_steps))
                print("lowest diff", lowest_diff)
                print("mom", mom_score, mom_position)
                print("dad", dad_score, dad_position)
                print("worst", worst)
                print("best vector", best_vector)
                letters = ['#', '%', "'", '(', ')', '*', '-', '.', '3', '7', '8', '9', ';', '>', '?', 'A', 'B', 'C', 'E', 'F', 'K', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'W', '[', '\\', ']', 'a', 'b', 'c', 'd', 'i', 'j', 'k', 'm', 'n', 'o', 'p', 'r', 's', 'u', 'v', 'w', 'y']
                # bv = [[letters[i], e] for i, e in enumerate(best_vector)]
                # print("best vector", bv)
                print()
        with open("best_masks_elite.txt", "a") as f:
            f.write(str(best_vector) + " - " + str(lowest_diff) + "\n")
        return best_vector


class EDA:
    def __init__(self,  crossover='uniform', num_steps=50000, pop_size=100, mutation_rate=0.05, ONE_RATE = 0.5):
        self.crossover = crossover
        self.num_steps = num_steps
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.ONE_RATE = ONE_RATE
    
    def evolve(self, x, mask, eval_func, classifier,goal, trial, verbose=False):
        population = []
        best_vector = None
        lowest_diff = float("inf")
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        for _ in range(self.pop_size):
            population.append(np.random.normal(mean, std))
        with open("logs/eda/" + str(trial) + "eda.txt", "w") as f:
            f.write("\n")
        for i, p in enumerate(population):
            newp = []
            for j, e in enumerate(p):
                if mask[j]:
                    newp.append(e)
            acc = eval_func(classifier, goal, np.array(newp))
            # print(newp)
            population[i] = [np.array(newp), acc]
        # pprint(population)
        for step in range(self.num_steps):
            population.sort(key=lambda x: x[1], reverse=True)
            mom_position, dad_position = choose_parents_tournament_lowest(population, 34)
            # print(population)
            # exit()
            worst = population[0][1]
            mom_score = population[mom_position][1]
            dad_score = population[dad_position][1]

            lowest_diff = population[len(population) - 1][1]
            best_vector = population[len(population) - 1][0]

            babies, inds = make_babies_eda(population, 20, self.mutation_rate)
            for q in range(len(babies)):
                babies[q] = [babies[q], eval_func(classifier, goal, babies[q])]
            babies.sort(key=lambda x: x[1])
            for q, ind in enumerate(inds):
                population[ind] = babies[q]        
            with open("logs/eda/" + str(trial) + "eda.txt", "a") as f:
                f.write(",".join([str(step), str(lowest_diff),str(worst)]) + "\n")
            if step % 100 == 0:
                print("step " + str(step) + " / " + str(self.num_steps))
                print("lowest diff", lowest_diff)
                print("mom", mom_score, mom_position)
                print("dad", dad_score, dad_position)
                print("worst", worst)
                print("best vector", best_vector)
                # letters = ['#', '%', "'", '(', ')', '*', '-', '.', '3', '7', '8', '9', ';', '>', '?', 'A', 'B', 'C', 'E', 'F', 'K', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'W', '[', '\\', ']', 'a', 'b', 'c', 'd', 'i', 'j', 'k', 'm', 'n', 'o', 'p', 'r', 's', 'u', 'v', 'w', 'y']
                # bv = [[letters[i], e] for i, e in enumerate(best_vector)]
                # print("best vector", bv)
                print()
        with open("best_masks_eda.txt", "a") as f:
            f.write(str(best_vector) + " - " + str(lowest_diff) + "\n")
        return best_vector