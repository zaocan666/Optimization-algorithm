# Optimization-algorithm
智能优化算法大作业，使用GA和SA求解函数优化问题和TSP问题

##文件
- GA_algorithm：GA算法框架，被main_f_min.py和main_TSP.py调用
- main_f_min.py：函数优化问题主文件，可使用GA或SA算法，同时可以指定两种算法所用参数
- main_TSP.py：TSP问题主文件，同上

## 运行
运行main_f_min.py或main_TSP.py，可在命令行传入参数或在代码内改变参数默认值，来选择所用算法，算法参数，运行模式等。比如：

    python main_TSP.py --method GA --mood history --map_mood random --GA_N 50 --GA_C 0.95 --GA_M 0.02

表示求解TSP问题，使用GA算法，运行模式是显示效用值历史，地图生成方式是随机生成，GA算法参数为：--GA_N 50 --GA_C 0.95 --GA_M 0.02
