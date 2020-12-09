import numpy as np
import copy

class GA_optimizer():
    def __init__(self, class_individual, N, C, M, nochange_iter, history_convert = lambda x:x):
        # class_individual: 个体的class，可调用产生新个体
        # N: 种群规模
        # C: 交叉概率
        # M: 变异概率
        # nochange_iter: 性能最好的个体保持不变nochange_iter回合后，优化结束
        # history_convert: 在记录history时将fitness转化为实际效用
        assert N%2==0 # 默认种群规模为偶数
        self.class_individual = class_individual
        self.N = N
        self.C = C
        self.M = M
        self.nochange_iter = nochange_iter
        self.history_convert = history_convert

    # 由旧种群产生新种群，包含交叉变异操作
    def selection(self, population, fitnesses):
        # 按排名分配被选择的概率
        population_fit_sorted = sorted(zip(fitnesses, population), key=lambda x:x[0]) # 按fitnesses从小到大排序
        population_sorted = list(zip(*population_fit_sorted))[1]
        choose_probability = list(range(1, len(population_sorted)+1))
        choose_probability = np.array(choose_probability)/np.sum(choose_probability)
        new_population = [population_sorted[-1], population_sorted[-2]] # 先将当前种群效用最佳的两个个体继承到子代种群
        for i in range(self.N//2-2):
            # 按适配值大小随机选择两个个体
            p1 = np.random.choice(population_sorted, p=choose_probability)
            p2 = np.random.choice(population_sorted, p=choose_probability)
            
            # 按概率随机选择是否进行交叉
            if np.random.rand()<self.C:
                p1_chromosome_new, p2_chromosome_new = p1.crossover(p2)
                p1_new = self.class_individual(p1_chromosome_new)
                p2_new = self.class_individual(p2_chromosome_new)
            else:
                p1_new = copy.deepcopy(p1)
                p2_new = copy.deepcopy(p2)
            
            # 进行随机变异
            p1_new.mutation(self.M)
            p2_new.mutation(self.M)

            new_population.append(p1_new)
            new_population.append(p2_new)

        return new_population


    def optimize(self, max_iteration, verbose=True):
        # max_iteration: 最大迭代次数
        # verbose: 是否有print

        population = []
        for i in range(self.N):
            a = self.class_individual()
            a.randomize()
            population.append(a)

        best_individual = population[0] # 保优操作，保存性能最好的个体
        nochange_iter_running = self.nochange_iter
        fitness_history = []

        for i in range(max_iteration):
            fitness_history.append(self.history_convert(best_individual.fitness()))
            if nochange_iter_running < 0:
                break

            fitnesses = [a.fitness() for a in population]

            # 找到最优的个体，保优
            best_index = np.argmax(fitnesses)
            if fitnesses[best_index] > best_individual.fitness():
                nochange_iter_running = self.nochange_iter
                best_individual = copy.deepcopy(population[best_index])

            population = self.selection(population, fitnesses)
            
            nochange_iter_running = nochange_iter_running - 1

            if verbose:
                print('iteration: %d\t best_individual: %s\t best_fitness: %.3f\t nochange_iter:%d'%(i, best_individual.__repr__(), best_individual.fitness(), nochange_iter_running))
            
        return best_individual, fitness_history