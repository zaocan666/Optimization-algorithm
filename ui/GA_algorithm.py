import numpy as np
import copy
import random

# GA种群中的个体定义，函数自变量有两个
class GA_Individual():
    def __init__(self, num_of_points, chromosome=None):
        self.num_of_points = num_of_points
        if type(chromosome) == list:
            self.num_of_points = len(chromosome)
            self.chromosome = chromosome  # x0和x1的二进制代码
        else:
            self.chromosome = list(range(self.num_of_points))

    def randomize(self):
        np.random.shuffle(self.chromosome)

    # 单位置次序交叉
    def crossover_order(self, p2):
        # p2: 用于交叉的另一个个体
        p1_chromosome = self.chromosome
        p2_chromosome = p2.chromosome
        # 随机选择一个节点进行交叉
        joint = np.random.choice(range(1, self.num_of_points - 1))

        # 从chromosome_added中去掉chromosome_part的节点，然后将结果与chromosome_part拼接起来
        def complete_chromosome(chromosome_part, chromosome_added):
            # chromosome_part: 待补充的染色体
            # chromosome_added: 用于补充的染色体
            left_points = list(set(chromosome_added) - set(chromosome_part))
            left_points_sorted = sorted(left_points, key=chromosome_added.index)
            return chromosome_part + left_points_sorted

        p1_chromosome_new_part = p1_chromosome[:joint]
        p1_chromosome_new_complete = complete_chromosome(p1_chromosome_new_part, p2_chromosome)
        p2_chromosome_new_part = p2_chromosome[:joint]
        p2_chromosome_new_complete = complete_chromosome(p2_chromosome_new_part, p1_chromosome)
        return p1_chromosome_new_complete, p2_chromosome_new_complete

    # 部分映射交叉
    def crossover_partially(self, p2):
        # p2: 用于交叉的另一个个体
        p1_chromosome = self.chromosome
        p2_chromosome = p2.chromosome
        # 随机选择两个节点进行交叉
        joint_left = np.random.choice(range(self.num_of_points - 1))
        joint_right = np.random.choice(range(joint_left + 1, self.num_of_points))

        def conflict_fill_in(chromosome_base, chromosome_added):
            # chromosome_base: 待填入的染色体
            # chromosome_added: 用于填入的染色体
            result = []
            for i in range(len(chromosome_base)):
                if i >= joint_left and i < joint_right:
                    result.append(chromosome_added[i - joint_left])
                else:
                    point_base = chromosome_base[i]
                    while True:
                        if point_base in chromosome_added:
                            index_in_added = chromosome_added.index(point_base)
                            point_base = chromosome_base[index_in_added + joint_left]
                        else:
                            break
                    result.append(point_base)
            return result

        p1_chromosome_new = conflict_fill_in(p1_chromosome, p2_chromosome[joint_left:joint_right])
        p2_chromosome_new = conflict_fill_in(p2_chromosome, p1_chromosome[joint_left:joint_right])
        return p1_chromosome_new, p2_chromosome_new

    def crossover(self, p2):
        return self.crossover_partially(p2)
        # return self.crossover_order(p2)

    def mutation(self, p):
        if np.random.rand() > p:
            return
        index_list = list(range(len(self.chromosome)))
        sam = list(random.sample(index_list, 2))
        start, end = min(sam), max(sam)
        tmp = self.chromosome[start:end]
        # np.random.shuffle(tmp)
        tmp = tmp[::-1]
        self.chromosome[start:end] = tmp

    def fitness(self, TSP_map):
        distance_sum = TSP_map.route_distance(self.chromosome)
        return -distance_sum

    def __repr__(self):
        return str(self.chromosome)

class GA_optimizer():
    def __init__(self, TSP_map, N, C, M, nochange_iter, last_generation_left=0.2, choose_mode='range', history_convert = lambda x:x):
        # class_individual: 个体的class，可调用产生新个体
        # N: 种群规模
        # C: 交叉概率
        # M: 变异概率
        # nochange_iter: 性能最好的个体保持不变nochange_iter回合后，优化结束
        # history_convert: 在记录history时将fitness转化为实际效用
        # last_generation_left: 保留上一代的比例
        # choose_mode: range（按排名算概率）/ fitness（按fitness算概率）

        self.TSP_map = TSP_map
        self.class_individual = GA_Individual
        self.N = N
        self.C = C
        self.M = M
        self.nochange_iter = nochange_iter
        self.history_convert = history_convert
        self.last_generation_left=last_generation_left
        self.choose_mode = choose_mode

        self.population = []
        for i in range(self.N):
            a = self.class_individual(self.TSP_map.num_of_points)
            a.randomize()
            self.population.append(a)
        
        self.best_individual = self.population[0] # 保优操作，保存性能最好的个体
        self.nochange_iter_running = self.nochange_iter
        self.fitness_history = []

    # 由旧种群产生新种群，包含交叉变异操作
    def selection(self, population, fitnesses):
        # 按排名分配被选择的概率
        population_fit_sorted = sorted(zip(fitnesses, population), key=lambda x:x[0]) # 按fitnesses从小到大排序
        population_sorted = list(zip(*population_fit_sorted))[1]
        population_sorted_left = population_sorted[::-1][:int(self.last_generation_left*len(population_sorted))]
        population_sorted_left = population_sorted_left[::-1]
        if self.choose_mode=='range':
            choose_probability = list(range(1, len(population_sorted_left)+1))
            choose_probability = np.array(choose_probability)/np.sum(choose_probability)
        elif self.choose_mode=='fitness':
            fitness_sorted = list(zip(*population_fit_sorted))[0]
            choose_probability = (fitness_sorted[::-1][:len(population_sorted_left)])[::-1]
            choose_probability = np.array(choose_probability)/np.sum(choose_probability)

        new_population = [population_sorted_left[-1], population_sorted_left[-2]] # 先将当前种群效用最佳的两个个体继承到子代种群
        # new_population = []
        while len(new_population)<self.N:
            # 按适配值大小随机选择两个个体
            p1 = np.random.choice(population_sorted_left, p=choose_probability)
            p2 = np.random.choice(population_sorted_left, p=choose_probability)
            
            # 按概率随机选择是否进行交叉
            if np.random.rand()<self.C:
                p1_chromosome_new, p2_chromosome_new = p1.crossover(p2)
                p1_new = self.class_individual(self.TSP_map.num_of_points, p1_chromosome_new)
                p2_new = self.class_individual(self.TSP_map.num_of_points, p2_chromosome_new)
            else:
                p1_new = copy.deepcopy(p1)
                p2_new = copy.deepcopy(p2)
            
            # 进行随机变异
            p1_new.mutation(self.M)
            p2_new.mutation(self.M)

            if p1_new.fitness(self.TSP_map)>p2_new.fitness(self.TSP_map):
                new_population.append(p1_new)
            else:
                new_population.append(p2_new)

        return new_population

    def step(self, verbose=True):
        # verbose: 是否有print

        self.fitness_history.append(self.history_convert(self.best_individual.fitness(self.TSP_map)))
        if self.nochange_iter_running < 0:
            return False

        fitnesses = [a.fitness(self.TSP_map) for a in self.population]

        # 找到最优的个体，保优
        best_index = np.argmax(fitnesses)
        if fitnesses[best_index] > self.best_individual.fitness(self.TSP_map):
            self.nochange_iter_running = self.nochange_iter
            self.best_individual = copy.deepcopy(self.population[best_index])

        self.population = self.selection(self.population, fitnesses)
        
        self.nochange_iter_running = self.nochange_iter_running - 1

        if verbose:
            print('best_individual: %s\t best_fitness: %.3f\t nochange_iter:%d'%(self.best_individual.__repr__(), self.history_convert(self.best_individual.fitness(self.TSP_map)), self.nochange_iter_running))
        
        return self.best_individual
    