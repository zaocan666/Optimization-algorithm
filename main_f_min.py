import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

from GA_algorithm import GA_optimizer

def aim_function(xs):
    return xs[1]*math.sin(xs[0])+xs[0]*math.cos(xs[1])

# GA种群中的个体定义，函数自变量有两个
class GA_Individual():
    xs_bounds = [[-10, 10], [-10, 10]] # x0, x1取值范围
    len_of_bin = 16 # x0和x1均用16位二进制数表示，可精确到小数点后3位
    def __init__(self, chromosome=None):
        if type(chromosome)==str:
            self.chromosome =  chromosome # x0和x1的二进制代码
        else:
            self.chromosome = '0'*(self.len_of_bin*2)

    def bin2num(self, bin_str, xi):
        return self.xs_bounds[xi][0] + int(bin_str, 2)/float((2**(self.len_of_bin))-1)*(self.xs_bounds[xi][1]-self.xs_bounds[xi][0]) # a+[S]_2/(2^L-1)*(b-a)

    def xs_chromosome(self, chromosome=None):
        if chromosome:
            x0 = chromosome[:self.len_of_bin]
            x1 = chromosome[self.len_of_bin:]
        else:
            x0 = self.chromosome[:self.len_of_bin]
            x1 = self.chromosome[self.len_of_bin:]
        return x0, x1

    def randomize(self):
        chromosome_int = np.random.randint(0, 2, 2*self.len_of_bin)
        chromosome = ''.join(map(str, chromosome_int))
        self.chromosome = chromosome

    def crossover(self, p2):
        # p2: 用于交叉的另一个个体
        p1_chromosome = self.chromosome
        p2_chromosome = p2.chromosome
        # 随机选择两个节点进行交叉
        joint_left = np.random.choice(range(self.len_of_bin))
        joint_right = np.random.choice(range(self.len_of_bin, self.len_of_bin*2))
        p1_chromosome_new = p1_chromosome[:joint_left] + p2_chromosome[joint_left:joint_right] + p1_chromosome[joint_right:]
        p2_chromosome_new = p2_chromosome[:joint_left] + p1_chromosome[joint_left:joint_right] + p2_chromosome[joint_right:]
        return p1_chromosome_new, p2_chromosome_new

    def mutation(self, p):
        change_prob = np.random.rand(self.len_of_bin*2) # 0到1之间的均匀分布随机采样
        change_flag = change_prob < p
        change_flag_str = ''.join(list(map(lambda x:str(int(x)), change_flag))) # 随机采样结果中小于p的位为1，大于p的位为0
        changed_chromosome = ('{:0%db}'%(2*self.len_of_bin)).format(int(change_flag_str, 2) ^ int(self.chromosome, 2)) # chromosome中对应change_flag_str里为1的位改变，为0的位不变
        self.chromosome = changed_chromosome

    def fitness(self):
        x0_str = self.chromosome[:self.len_of_bin]
        x1_str = self.chromosome[self.len_of_bin:]
        return aim_function([self.bin2num(x0_str, 0), self.bin2num(x1_str, 1)])

    def __repr__(self):
        x0, x1 = self.xs_chromosome()
        return "x0: %.3f x1:%.3f"%(self.bin2num(x0, 0), self.bin2num(x1, 1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='GA', help='type of optimization method')
    parser.add_argument('--max_iteration', type=int, default=2000, help='type of optimization method')
    parser.add_argument('--random_seed', type=int, default=-1, help='type of optimization method')
    parser.add_argument('--mood', type=str, default='history', help='once/history/multi_times')
    # GA param
    parser.add_argument('--GA_N', type=int, default=30, help='size of population')
    parser.add_argument('--GA_C', type=float, default=0.95, help='probability of crossover')
    parser.add_argument('--GA_M', type=float, default=0.1, help='probability of mutaion')
    parser.add_argument('--GA_nochange_iter', type=int, default=100, help='num of iteration without change of best individual before stop')
    # SA param

    args = parser.parse_args()

    # 如果args.random_seed>=0，则设置随机数种子
    if args.random_seed>=0:
        np.random.seed(args.random_seed)

    if args.method == 'GA':
        optimizer = GA_optimizer(GA_Individual, args.GA_N, args.GA_C, args.GA_M, args.GA_nochange_iter)

    # 绘制出某次仿真过程中目标函数的变化曲线
    if args.mood=='history':
        _, fitness_history = optimizer.optimize(args.max_iteration)
        plt.plot(fitness_history)
        plt.xlabel('iteration')
        plt.ylabel('f(X)')
        plt.show()
    
    # 给出20次随机实验的统计结果（平均性能、最佳性能、最差性能、方差等）
    elif args.mood=='multi_times':
        best_fitnesses = []
        for i in range(20):
            best_individual, _ = optimizer.optimize(args.max_iteration, verbose=False)
            best_fitnesses.append(best_individual.fitness())
            print(best_fitnesses[-1])
        best_fitnesses = np.array(best_fitnesses)
        print('平均性能: %.3f\t 最佳性能: %.3f\t 最差性能: %.3f\t 方差: %.8f'%
            (best_fitnesses.mean(), best_fitnesses.max(), best_fitnesses.min(), best_fitnesses.var()))
        
        plt.plot(best_fitnesses)
        plt.xlabel(u'实验次数')
        plt.ylabel(u'实验性能')
        plt.show()

    # 一次优化结果
    elif args.mood=='once':
        optimizer.optimize(args.max_iteration)

