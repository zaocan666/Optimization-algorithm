import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

from GA_algorithm import GA_optimizer

class TSP_MAP():
    def __init__(self, num_of_points=0):
        self.num_of_points = num_of_points

    # 计算距离矩阵 distance_martix[i, j] = distance(points[i], points[j])
    def calculate_distance_martix(self):
        G = np.dot(self.points, self.points.T) #[n, n]
        H = np.tile(np.diag(G), (self.points.shape[0],1))
        distance_martix = H + H.T - 2*G
        self.distance_martix = np.power(distance_martix, 0.5)

    def random_generate(self):
        self.points = np.random.rand(self.num_of_points, 2)*100
        self.calculate_distance_martix()

    def read_from_file(self, file_route):
        with open(file_route, 'r') as f:
            lines = f.readlines()
        self.num_of_points = int(lines[0].strip())
        points = []
        for line in lines[1:]:
            ps = line.strip().split(' ')
            points.append([float(ps[0]), float(ps[1])])
        
        self.points = np.array(points)
        self.calculate_distance_martix()

    def route_distance(self, route):
        distance_sum = 0
        for i in range(self.num_of_points-1):
            distance_sum += self.distance_martix[route[i], route[i+1]]
        distance_sum += self.distance_martix[route[-1], route[0]]

        return distance_sum
    
    def draw_route(self, route):
        plt.scatter(self.points[:,0], self.points[:,1], s=3, c='red')
        for i in range(self.num_of_points-1):
            plt.plot([self.points[route[i],0], self.points[route[i+1],0]], [self.points[route[i],1], self.points[route[i+1],1]])
        plt.plot([self.points[route[-1],0], self.points[route[0],0]], [self.points[route[-1],1], self.points[route[0],1]])

# GA种群中的个体定义，函数自变量有两个
class GA_Individual():
    def __init__(self, chromosome=None):
        self.num_of_points = TSP_map.num_of_points
        if type(chromosome)==list:
            self.chromosome =  chromosome # x0和x1的二进制代码
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
        joint = np.random.choice(range(1, self.num_of_points-1))

        # 从chromosome_added中去掉chromosome_part的节点，然后将结果与chromosome_part拼接起来
        def complete_chromosome(chromosome_part, chromosome_added):
            # chromosome_part: 待补充的染色体
            # chromosome_added: 用于补充的染色体
            left_points = list(set(chromosome_added)-set(chromosome_part))
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
        joint_left = np.random.choice(range(self.num_of_points-1))
        joint_right = np.random.choice(range(joint_left+1, self.num_of_points))

        def conflict_fill_in(chromosome_base, chromosome_added):
            # chromosome_base: 待填入的染色体
            # chromosome_added: 用于填入的染色体
            result = []
            for i in range(len(chromosome_base)):
                if i>=joint_left and i<joint_right:
                    result.append(chromosome_added[i-joint_left])
                else:
                    point_base = chromosome_base[i]
                    while True:
                        if point_base in chromosome_added:
                            index_in_added = chromosome_added.index(point_base)
                            point_base = chromosome_base[index_in_added+joint_left]
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
        for i in range(self.num_of_points):
            if np.random.rand()>p:
                continue
            other_part = list(set(range(self.num_of_points))-set([i]))
            change_j = np.random.choice(other_part)
            temp = self.chromosome[change_j]
            self.chromosome[change_j] = self.chromosome[i]
            self.chromosome[i] = temp

    def fitness(self):
        distance_sum = TSP_map.route_distance(self.chromosome)
        return -distance_sum

    def __repr__(self):
        return str(self.chromosome)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='GA', help='type of optimization method')
    parser.add_argument('--max_iteration', type=int, default=5000, help='type of optimization method')
    parser.add_argument('--random_seed', type=int, default=-1, help='type of optimization method')
    parser.add_argument('--mood', type=str, default='multi_times', help='task index, value:[once/history/multi_times]')
    parser.add_argument('--map_mood', type=str, default='read', help='way of getting map, value: [random/read]')
    parser.add_argument('--map_point_num', type=int, default=30, help='num of points in the TSP map')
    parser.add_argument('--map_file', type=str, default='TSP_points/BEN30-XY.txt', help='route of map points file')
    # GA param
    parser.add_argument('--GA_N', type=int, default=60, help='size of population')
    parser.add_argument('--GA_C', type=float, default=0.95, help='probability of crossover')
    parser.add_argument('--GA_M', type=float, default=0.01, help='probability of mutaion')
    parser.add_argument('--GA_nochange_iter', type=int, default=500, help='num of iteration without change of best individual before stop')
    # SA param

    args = parser.parse_args()

    if args.map_mood == 'random':
        TSP_map = TSP_MAP(args.map_point_num)
        TSP_map.random_generate()
    elif args.map_mood == 'read':
        TSP_map = TSP_MAP()
        TSP_map.read_from_file(args.map_file)

    # 如果args.random_seed>=0，则设置随机数种子
    if args.random_seed>=0:
        np.random.seed(args.random_seed)

    if args.method == 'GA':
        optimizer = GA_optimizer(GA_Individual, args.GA_N, args.GA_C, args.GA_M, args.GA_nochange_iter, history_convert=lambda x:-x)

    # 绘制出某次仿真过程中目标函数的变化曲线
    if args.mood=='history':
        best_individual, fitness_history = optimizer.optimize(args.max_iteration)
        plt.plot(fitness_history)
        plt.xlabel('iteration')
        plt.ylabel('f(X)')

        plt.figure()
        TSP_map.draw_route(best_individual.chromosome)

        plt.show()
    
    # 给出20次随机实验的统计结果（平均性能、最佳性能、最差性能、方差等）
    elif args.mood=='multi_times':
        best_fitnesses = []
        for i in range(20):
            best_individual, _ = optimizer.optimize(args.max_iteration, verbose=False)
            best_fitnesses.append(-best_individual.fitness())
            print(best_fitnesses[-1])
        best_fitnesses = np.array(best_fitnesses)
        print('平均性能: %.3f\t 最佳性能: %.3f\t 最差性能: %.3f\t 方差: %.8f'%
            (best_fitnesses.mean(), best_fitnesses.min(), best_fitnesses.max(), best_fitnesses.var()))
        
        plt.plot(best_fitnesses)
        plt.xlabel(u'实验次数')
        plt.ylabel(u'实验性能')
        plt.show()

    # 一次优化结果
    elif args.mood=='once':
        optimizer.optimize(args.max_iteration)

