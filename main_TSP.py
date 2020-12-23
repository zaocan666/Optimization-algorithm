import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import time
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

from GA_algorithm import GA_optimizer


class TSP_MAP():
    def __init__(self, num_of_points=0):
        self.num_of_points = num_of_points

    # 计算距离矩阵 distance_martix[i, j] = distance(points[i], points[j])
    def calculate_distance_martix(self):
        G = np.dot(self.points, self.points.T)  # [n, n]
        H = np.tile(np.diag(G), (self.points.shape[0], 1))
        distance_martix = H + H.T - 2 * G
        self.distance_martix = np.power(distance_martix, 0.5)

    def random_generate(self):
        self.points = np.random.rand(self.num_of_points, 2) * 100
        self.calculate_distance_martix()

    def read_from_file(self, file_route):
        with open(file_route, 'r') as f:
            lines = f.readlines()
        self.num_of_points = int(lines[0].strip())
        points = []
        for line in lines[1:]:
            try:
                ps = line.strip().split(' ')
                points.append([float(ps[0]), float(ps[1])])
            except:
                break

        self.points = np.array(points)
        self.calculate_distance_martix()

    def route_distance(self, route):
        distance_sum = 0
        for i in range(self.num_of_points - 1):
            distance_sum += self.distance_martix[route[i], route[i + 1]]
        distance_sum += self.distance_martix[route[-1], route[0]]

        return distance_sum

    def draw_route(self, route):
        plt.scatter(self.points[:, 0], self.points[:, 1], s=3, c='red')
        for i in range(self.num_of_points - 1):
            plt.plot([self.points[route[i], 0], self.points[route[i + 1], 0]],
                     [self.points[route[i], 1], self.points[route[i + 1], 1]])
        plt.plot([self.points[route[-1], 0], self.points[route[0], 0]],
                 [self.points[route[-1], 1], self.points[route[0], 1]])


# GA种群中的个体定义，函数自变量有两个
class GA_Individual():
    def __init__(self, chromosome=None):
        self.num_of_points = TSP_map.num_of_points
        if type(chromosome) == list:
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

    def fitness(self):
        distance_sum = TSP_map.route_distance(self.chromosome)
        return -distance_sum

    def __repr__(self):
        return str(self.chromosome)


class SA_TSP():
    def __init__(self,TSP_MAP,T0_mode='random',route_mode='SWAP',T_annealing_mode = 'log'):
        # TSP_MAP：TSP地图
        # T0_mode: 初温生成方式
        # route_mode: 路径生成方式
        # T_annealing_mode: 退火模式

        self.TSP_MAP = TSP_MAP
        self.points = TSP_MAP.points
        self.N = TSP_map.num_of_points
        self.T0_mode = T0_mode
        self.route_mode = route_mode
        self.T_annealing_mode = T_annealing_mode

    def init_outer_para(self,T_converge_mode='step',T_Lambda=0.05,T_end = 0.5,T_out_step=100,T_out_dE_step=100,T_out_dE_threshold=5):
        # 外循环参数
        # T_converge_mode: 外循环收敛方式
        # T_Lambda: 温度衰减系数
        # T_end: 基于温度的收敛--终止温度
        # T_out_step: 基于迭代次数的收敛--外循环中温度的迭代次数
        # T_out_dE_step: 基于性能的收敛--搜索到的最优值连续若干步变化微小
        # T_out_dE_threshold: 基于性能的收敛--搜索到的最优值连续若干步变化微小

        self.T_converge_mode = T_converge_mode
        self.T_Lambda = T_Lambda
        self.T_end = T_end
        self.T_out_step = T_out_step
        self.T_out_dE_step = T_out_dE_step
        self.T_out_dE_threshold = T_out_dE_threshold

    def init_inner_para(self,T_in_step=100,T_in_threshold='threshold',T_Metropolis_mode='threshold'):
        # 内循环参数
        # T_Metropolis: 内循环模式（'threshold':连续若干步的目标值变化小于预设阈值/'step':固定步数抽样)
        # T_in_step：基于迭代次数的收敛--内循环中每个温度下搜索的次数
        # T_in_threshold: 基于性能的收敛--内循环中相邻两个目标函数之差小于T_in_threshold时认为趋向于收敛

        self.T_Metropolis_mode = T_Metropolis_mode
        self.T_in_step = T_in_step
        self.T_in_threshold = T_in_threshold

    def init_T0(self,T0_mode):
        # T0_mode: 初温生成模式
        # experience: 根据经验设定初温
        # random: 均匀随机产生一组状态，确定两两状态间的最大目标值差，设定最差状态相对最佳状态的接受概率p=0.9

        if T0_mode =='experience':
            return 700

        elif T0_mode =='random':
            dest_value = []
            for i in range(0,50):
                route = random.sample(range(0,self.N),self.N)
                dest_value.append(self.TSP_MAP.route_distance(route))
            return ( max(dest_value)-min(dest_value) )*1.0/abs(math.log(0.9))

    def new_state(self, route):
        # route_mode: 更新TSP路径时使用的模式
        # SWAP: 指定两位置元素交换
        # REVERSE：指定两位置之间的元素逆序
        # INSERT: 把指定某一段序列插到第三个位置之后
        # MULTI: 按不同概率把上述操作混合
        if self.route_mode == 'SWAP':
            while 1:
                index1, index2 = np.random.randint(0, self.N, 2)
                if index1 != index2:
                    break
            route[index1], route[index2] = route[index2], route[index1]
        elif self.route_mode == 'REVERSE':
            while 1:
                index1, index2 = np.random.randint(0, self.N, 2)
                if index1 != index2:
                    break
            reverse_str = copy.deepcopy(route[index1:index2][::-1])
            route[index1:index2] = reverse_str
        elif self.route_mode == 'INSERT':
            while 1:
                index1, index2, index3 = np.random.randint(0, self.N, 3)  # >=0, < self.N
                if index1 != index2 and index1 != index3 and index2 != index3:
                    break
            if index1 > index2:
                index1, index2 = index2, index1
            if index2 > index3:
                index2, index3 = index3, index2
            if index1 > index2:
                index1, index2 = index2, index1
            temp_str = copy.deepcopy(route[index1:index2])
            route[index1:index1 + (index3 - index2 + 1)] = route[index2:index3 + 1]
            route[index1 + (index3 - index2 + 1):index3 + 1] = temp_str
        elif self.route_mode == 'MULTI1':
            if random.random()<0.5:
                while 1:
                    index1, index2 = np.random.randint(0, self.N, 2)
                    if index1 != index2:
                        break
                route[index1], route[index2] = route[index2], route[index1]
            else:
                while 1:
                    index1, index2, index3 = np.random.randint(0, self.N, 3)  # >=0, < self.N
                    if index1 != index2 and index1 != index3 and index2 != index3:
                        break
                if index1 > index2:
                    index1, index2 = index2, index1
                if index2 > index3:
                    index2, index3 = index3, index2
                if index1 > index2:
                    index1, index2 = index2, index1
                temp_str = copy.deepcopy(route[index1:index2])
                route[index1:index1 + (index3 - index2 + 1)] = route[index2:index3 + 1]
                route[index1 + (index3 - index2 + 1):index3 + 1] = temp_str
        elif self.route_mode == 'MULTI2':
            random_num = random.random()
            if random_num <0.2:
                while 1:
                    index1, index2 = np.random.randint(0, self.N, 2)
                    if index1 != index2:
                        break
                route[index1], route[index2] = route[index2], route[index1]
            elif random_num >=0.2 and random_num<0.6:
                while 1:
                    index1, index2, index3 = np.random.randint(0, self.N, 3)  # >=0, < self.N
                    if index1 != index2 and index1 != index3 and index2 != index3:
                        break
                if index1 > index2:
                    index1, index2 = index2, index1
                if index2 > index3:
                    index2, index3 = index3, index2
                if index1 > index2:
                    index1, index2 = index2, index1
                temp_str = copy.deepcopy(route[index1:index2])
                route[index1:index1 + (index3 - index2 + 1)] = route[index2:index3 + 1]
                route[index1 + (index3 - index2 + 1):index3 + 1] = temp_str
            else:
                while 1:
                    index1, index2 = np.random.randint(0, self.N, 2)
                    if index1 != index2:
                        break
                reverse_str = copy.deepcopy(route[index1:index2][::-1])
                route[index1:index2] = reverse_str
        return route


    def annealing(self,T,k_step,T_Lambda,T_annealing_mode):
        # T_annealing_mode: 退火模式(ordinary常用指数退温,log即温度与退温不输的对数成反比)
        #指数退温和温度与退温步数的对数成反比
        if T_annealing_mode == 'ordinary':
            T = T_Lambda * T
        elif T_annealing_mode == 'log':
            T = T*1.0/math.log(1+k_step)
        return T


    def state_accept_p(self,E_0,E_1,t):
        # p: 从状态E(n)转移到E(n+1)的概率
        # p=1: 接受状态转移
        # p<1: 产生[0,1]随机数，若小于p则转移，否则不转移
        # E_0/E_1: E(n)/E(n+1)

        p = min(1,np.exp(-(E_1-E_0)*1.0/t))
        #print('p:'+str(p))
        if random.random()<p:
            return True
        else:
            return False

    def optimize(self):
        T = self.init_T0(self.T0_mode)

        route_curr = np.arange(self.N)#random.sample(range(0,self.N),self.N)
        route_new = route_curr
        Distances = []
        D_min = self.TSP_MAP.route_distance(route_curr)

        best_route = route_curr
        D_curr = D_min
        k_step = 0
        out_break_times = 0
        Distances.append(D_min)

        while 1:
            # 外循环收敛准则
            if self.T_converge_mode == 'temperature':    # 基于时间的收敛：温度低于阈值
                if T < self.T_end:
                    break
            elif self.T_converge_mode == 'iteration':    # 基于时间的收敛：迭代次数高于阈值
                if k_step >= self.T_out_step:
                    break
            elif self.T_converge_mode == 'performance' : # 基于性能的收敛：搜索到的最优值连续若干步变化微小（会不会可能跳不出来啊
                if out_break_times >= self.T_out_dE_step :
                    break

            # 退温步数
            k_step += 1
            in_break_times = 0

            # 在每个温度下重新寻找最优解
            in_step = 0
            while 1:
                # 每个温度下执行T_iter_step个循环
                # 更新状态；计算新的目标函数值(<0就转移到新状态，否则按照一定概率转移);
                in_step += 1
                route_new = self.new_state(route_new)
                D_new = self.TSP_MAP.route_distance(route_new)

                # 状态转移
                if D_new < D_curr:
                    route_curr = route_new.copy()
                    D_curr = D_new
                if D_new < D_min:
                    best_route = route_new.copy()
                    D_min = D_new
                else:
                    if np.random.rand() < np.exp(-(D_new - D_curr) / (T)):
                        route_curr = route_new.copy()
                        D_curr = D_new
                    else:
                        route_new = route_curr.copy()

                # Metropolis抽样稳定准则:若连续若干步的目标函数之差小于设定阈值/循环步数大于阈值，则跳出循环(会不会可能跳不出来啊。。。
                if self.T_Metropolis_mode == 'threshold':
                    if abs(D_new - D_curr)<self.T_in_threshold:
                        in_break_times += 1
                        if in_break_times>= int(self.T_in_step/5):
                            break
                    else:
                        in_break_times = 0
                elif self.T_Metropolis_mode == 'step':
                    if in_step >= self.T_in_step:
                        break

            # 记录最优解随温度的变化，每个温度下记录一个最优解
            if k_step>1 and abs(D_min-Distances[-1]) < self.T_out_dE_threshold:
                out_break_times += 1
            else:
                out_break_times = 0

            Distances.append(D_min)#,k_step,T])
            T = self.annealing(T,k_step,self.T_Lambda,self.T_annealing_mode)

        return best_route, Distances

if __name__ == '__main__':
    T0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='SA', help='type of optimization method')
    parser.add_argument('--max_iteration', type=int, default=2000, help='type of optimization method')
    parser.add_argument('--random_seed', type=int, default=-1, help='type of optimization method')
    parser.add_argument('--mood', type=str, default='history', help='task index, value:[once/history/multi_times]')
    parser.add_argument('--map_mood', type=str, default='read', help='way of getting map, value: [random/read]')
    parser.add_argument('--map_point_num', type=int, default=30, help='num of points in the TSP map')
    parser.add_argument('--map_file', type=str, default='TSP_points/BEN75-XY.txt', help='route of map points file')
    # GA param
    parser.add_argument('--GA_N', type=int, default=60, help='size of population')
    parser.add_argument('--GA_C', type=float, default=0.95, help='probability of crossover')
    parser.add_argument('--GA_M', type=float, default=0.2, help='probability of mutaion')
    parser.add_argument('--GA_nochange_iter', type=int, default=500,
                        help='num of iteration without change of best individual before stop')
    parser.add_argument('--GA_last_gl', type=float, default=0.2, help='proportion of last generation left')
    # SA param
    parser.add_argument('--SA_T0_mode', type=str, default='experience', help='way of initializing temperature , value: [random/experience]')
    parser.add_argument('--SA_route_mode', type=str, default='MULTI2',help='way of updating route, value: [SWAP/REVERSE/INSERT/MULTI1/MULTI2]')
    parser.add_argument('--SA_T_annealing_mode', type=str, default='ordinary',help='way of annealing , value: [log/ordinary]')

    parser.add_argument('--SA_T_converge_mode', type=str, default='temperature',help='way of converging in the outer cycle , value: [temperature/iteration/performance]')
    parser.add_argument('--SA_T_Lambda', type=float, default=0.9,help='ratio of annealing')
    parser.add_argument('--SA_T_end', type=float, default=1e-5, help='minimum temperature in the outer cycle')
    parser.add_argument('--SA_T_out_step', type=int, default=200, help='maximum steps in the outer cycle')
    parser.add_argument('--SA_T_out_dE_step', type=int, default=300, help='maximum continuous steps of small change in the outer cycle')
    parser.add_argument('--SA_T_out_dE_threshold', type=int, default=21,help='threshold of difference between Distances(n) and Distances(n+1) in the outer cycle')

    parser.add_argument('--SA_T_Metropolis_mode', type=str, default='step',help='way of converging in the inner cycle , value: [step/threshold]')
    parser.add_argument('--SA_T_in_step', type=int, default=5000, help='maximum steps in the inner cycle')
    parser.add_argument('--SA_T_in_threshold', type=float, default=50,help='threshold of difference between f(n) and f(n+1) in the inner cycle')

    args = parser.parse_args()

    if args.map_mood == 'random':
        TSP_map = TSP_MAP(args.map_point_num)
        TSP_map.random_generate()
    elif args.map_mood == 'read':
        TSP_map = TSP_MAP()
        TSP_map.read_from_file(args.map_file)

    # 如果args.random_seed>=0，则设置随机数种子
    if args.random_seed >= 0:
        np.random.seed(args.random_seed)

    if args.method == 'GA':
        optimizer = GA_optimizer(GA_Individual, args.GA_N, args.GA_C, args.GA_M, 
                    args.GA_nochange_iter, choose_mode='range', last_generation_left=args.GA_last_gl, history_convert=lambda x: -x)
    elif args.method == 'SA':
        optimizer = SA_TSP(TSP_map, args.SA_T0_mode, args.SA_route_mode, args.SA_T_annealing_mode)
        optimizer.init_outer_para(args.SA_T_converge_mode, args.SA_T_Lambda, args.SA_T_end, args.SA_T_out_step, args.SA_T_out_dE_step, args.SA_T_out_dE_threshold)
        optimizer.init_inner_para(args.SA_T_in_step, args.SA_T_in_threshold, args.SA_T_Metropolis_mode)

    # 绘制出某次仿真过程中目标函数的变化曲线
    if args.mood == 'history':
        if args.method == 'GA':
            best_individual, fitness_history = optimizer.optimize(args.max_iteration)
            plt.plot(fitness_history)
            plt.xlabel('iteration')
            plt.ylabel('f(X)')
            end_time = time.time() - T0

            plt.figure()
            TSP_map.draw_route(best_individual.chromosome)

        elif args.method == 'SA':
            best_route, fitness_history = optimizer.optimize()
            plt.plot(fitness_history)
            plt.xlabel('iteration')
            plt.ylabel('f(X)')
            end_time = time.time() - T0

            plt.figure()
            TSP_map.draw_route(best_route)
        plt.title(args.method+'单次实验最短距离: %.3f 总耗时(不包括绘图): %.3f 包括绘图: %.3f'%
                  (fitness_history[-1], end_time,time.time()-T0))
        plt.show()

    # 给出20次随机实验的统计结果（平均性能、最佳性能、最差性能、方差等）
    elif args.mood == 'multi_times':
        min_dist = 10000
        best_fitnesses = []
        TIMES = []
        t_before= time.time()-T0
        for i in range(20):
            t0 = time.time()
            if args.method == 'GA':
                best_individual, _ = optimizer.optimize(args.max_iteration, verbose=False)
                TIMES.append(time.time() - t0+t_before)
                best_fitnesses.append(-best_individual.fitness())
                curr_route = best_individual.chromosome
            elif args.method == 'SA':
                curr_route, fit_history = optimizer.optimize()
                TIMES.append(time.time() - t0+t_before)
                best_fitnesses.append(fit_history[-1])
            if best_fitnesses[-1]<min_dist:
                min_dist = best_fitnesses[-1]
                min_dist_route = curr_route
            print(best_fitnesses[-1])

        best_fitnesses = np.array(best_fitnesses)
        TIMES = np.array(TIMES)
        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(best_fitnesses)
        ax1.set(title=args.method + '+TSP性能--平均: %.3f\n 最佳: %.3f 最差: %.3f\n 方差: %.8f' %
                      (best_fitnesses.mean(), best_fitnesses.min(), best_fitnesses.max(), best_fitnesses.var()),
                ylabel='实验性能', xlabel='实验次数')

        ax2 = plt.subplot(1, 2, 2)
        ax2.plot(TIMES)
        ax2.set(title=args.method + '+TSP时间--平均: %.3f\n最佳: %.3f 最差: %.3f\n 方差: %.8f' %
                      (TIMES.mean(), TIMES.min(), TIMES.max(), TIMES.var()),
                ylabel='运行部分耗时/s', xlabel='实验次数')
        plt.figure()
        TSP_map.draw_route(min_dist_route)
        plt.show()

    # 一次优化结果
    elif args.mood == 'once':
        if args.mood == 'GA':
            optimizer.optimize(args.max_iteration)
        elif args.mood == 'SA':
            optimizer.optimize()

    # main