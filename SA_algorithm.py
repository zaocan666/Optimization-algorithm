import numpy as np
import copy
import random
import math

#Simulated Annealing
class SA_optimizer():
    def __init__(self,T0_mode,T_annealing_mode,T_Lambda,x_eta,x_mode):
        # T0_mode: 初温生成模式
        # T_annealing_mode: 退火模式(ordinary常用指数退温,log即温度与退温不输的对数成反比)
        # T_Lambda: 当退火模式为指数退温时的温度衰减系数
        # scale_min: 自变量范围下限
        # scale_max: 自变量范围上限
        # x_eta: 自变量更新时的系数eta
        # x_mode: 更新自变量/状态时使用的模式（高斯分布Gauss/柯西分布Cauchy）

        self.T0_mode = T0_mode
        self.T_annealing_mode = T_annealing_mode
        self.T_Lambda = T_Lambda
        self.scale_max = 10
        self.scale_min = -10
        self.x_eta = x_eta
        self.x_mode = x_mode

    def init_outer_para(self,T_out_step,T_end,T_converge_mode,T_out_dE_threshold,T_out_dE_step):
        # T_out_step: 外循环中温度的迭代次数
        # T_end: 当外循环收敛模式为基于温度时，T_end为终止温度
        # T_converge_mode: 外循环温度的收敛准则

        self.T_out_step = T_out_step
        self.T_end = T_end
        self.T_converge_mode = T_converge_mode
        self.T_out_dE_threshold = T_out_dE_threshold
        self.T_out_dE_step = T_out_dE_step

    def init_inner_para(self,T_in_threshold,T_in_step,T_Metropolis_mode):
        # T_in_threshold: 相邻两个目标函数之差小于T_in_threshold时认为趋向于收敛
        # T_in_step：在以迭代步数为标准的内循环里，每个温度下的最大迭代次数
        # T_Metropolis_mode: 内循环收敛模式（'threshold':连续若干步的目标值变化小于预设阈值/'step':固定步数抽样)

        self.T_in_threshold = T_in_threshold
        self.T_in_step = T_in_step
        self.T_Metropolis_mode = T_Metropolis_mode

    def init_T0(self):
        # experience: 根据经验设定初温
        # unifrom: 均匀抽样一组状态，以各状态目标值的方差为初温
        # random: 均匀随机产生一组状态，确定两两状态间的最大目标值差，设定最差状态相对最佳状态的接受概率p=0.5

        if self.T0_mode=='experience':
            return 200

        elif self.T0_mode=='uniform':
            dest_value = []
            increment = (self.scale_max-self.scale_min)/10
            for i in range(0,11):
                x0 = self.scale_min + i * increment
                x1 = x0
                dest_value.append(self.function(x0, x1))
            return np.var(dest_value)

        elif self.T0_mode=='random':
            dest_value = []
            for i in range(0,50):
                x0 = (self.scale_max-self.scale_min) * np.random.randn()
                x1 = (self.scale_max-self.scale_min) * np.random.randn()
                [x0,x1] = [max( min(self.scale_max, x0), self.scale_min), max( min(self.scale_max, x1), self.scale_min)]
                dest_value.append(self.function(x0,x1))
            return ( max(dest_value)-min(dest_value) )*1.0/abs(math.log(0.5))

    def function(self,x0,x1):
        return x1 * math.sin(x0) + x0 * math.cos(x1)

    def state_accept_p(self,E_0,E_1,t):
        # p: 从状态E(n)转移到E(n+1)的概率
        # p=1: 接受状态转移
        # p<1: 产生[0,1]随机数，若小于p则转移，否则不转移
        # E_0/E_1: E(n)/E(n+1)

        p = min(1, math.exp(-(E_1-E_0)*1.0/t))
        if random.random()<p:
            return true
        else:
            return false

    def new_state(self,x0,x1):
        if self.x_mode=='Gauss':
            while 1:
                x0_ = x0 + self.x_eta * np.random.randn()
                x1_ = x1 + self.x_eta * np.random.randn()
                if (x0_ >= self.scale_min and x0_<=self.scale_max) and (x1_ >= self.scale_min and x1_<=self.scale_max):
                    break
        elif self.x_mode=='Cauchy':
            while 1:
                x0_ = x0 + self.x_eta * np.random.standard_cauchy()
                x1_ = x1 + self.x_eta * np.random.standard_cauchy()
                if (x0_ >= self.scale_min and x0_ <= self.scale_max) and (x1_ >= self.scale_min and x1_ <= self.scale_max):
                    break

        print(str(x0_)+','+str(x1_))
        return x0_,x1_

    def annealing(self,T,k_step):
        #指数退温和温度与退温步数的对数成反比

        if self.T_annealing_mode == 'ordinary':
            return self.T_Lambda * T
        elif self.T_annealing_mode == 'log':
            return T*1.0/math.log(1+k_step)

    def optimize(self):
        # 指定模式产生初温
        # 随机产生初始解x0,x1
        T = self.init_T0()
        x0, x1 = random.uniform(-10, 10), random.uniform(-10, 10)
        x0_new, x1_new = x0, x1
        x0_best, x1_best = x0, x1
        E_history = []
        E_min = self.function(x0, x1)
        E_current = E_min
        k_step = 0
        out_break_times = 0
        performance_flag = 1

        while 1:
            # 外循环收敛准则
            print('k_step:'+str(k_step))
            if self.T_converge_mode == 'temperature':    # 基于时间的收敛：温度低于阈值
                if T < self.T_end:
                    break
            elif self.T_converge_mode == 'iteration':    # 基于时间的收敛：迭代次数高于阈值
                if k_step >= self.T_out_step:
                    break
            elif self.T_converge_mode == 'performance' : # 基于性能的收敛：搜索到的最优值连续若干步变化微小（会不会可能跳不出来啊
                if out_break_times >= self.T_out_step and k_step < self.T_out_step:
                    break

            # 退温步数
            k_step += 1
            in_break_times = 0

            # 在每个温度下重新寻找最优解
            in_step = 0
            while 1:
                print('in_step:'+str(in_step))
                # 每个温度下执行T_iter_step个循环
                # 更新状态；计算新的目标函数值(<0就转移到新状态，否则按照一定概率转移);

                in_step += 1
                x0_new,x1_new = self.new_state(x0_new,x1_new)
                E_new = self.function(x0_new,x1_new)
                dE = E_new - E_current
                print('dE:'+str(dE))

                # 状态转移
                if dE<0:
                    E_current, x0, x1 = E_new, x0_new, x1_new
                if E_new < E_min:
                    E_min, x0_best, x1_best = E_new, x0_new, x1_new
                else:
                    if random.random() < math.exp(-dE/T):
                        E_current, x0, x1 = E_new, x0_new, x1_new
                    else:
                        x0_new, x1_new = x0, x1

                # Metropolis抽样稳定准则:若连续若干步的目标函数之差小于设定阈值，则跳出循环
                if self.T_Metropolis_mode == 'threshold':
                    if abs(dE)<self.T_in_threshold:
                        in_break_times += 1
                        if in_break_times>= int(self.T_in_step/3):
                            break
                    else:
                        in_break_times = 0
                elif self.T_Metropolis_mode == 'step':
                    if in_step >= self.T_in_step:
                        break

            # 记录最优解随温度的变化，每个温度下记录一个最优解
            if k_step>1 and abs(E_min-E_history[-1]) < self.T_out_dE_threshold:
                out_break_times += 1
            else:
                out_break_times = 0

            E_history.append(E_min)
            T = self.annealing(T,k_step)

        return x0_best, x1_best, E_history










