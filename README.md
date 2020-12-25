# Optimization-algorithm
智能优化算法大作业，使用GA和SA求解函数优化问题和TSP问题

##文件
- GA_algorithm：GA算法框架，被main_f_min.py和main_TSP.py调用
- SA_algorithm：SA算法框架，被main_f_min.py调用
- main_f_min.py：函数优化问题主文件，可使用GA或SA算法，同时可以指定两种算法所用参数
- main_TSP.py：TSP问题主文件，同上

## 运行
运行main_f_min.py或main_TSP.py，可在命令行传入参数或在代码内改变参数默认值，来选择所用算法，算法参数，运行模式等。比如：

    python main_TSP.py --method GA --mood history --map_mood random --GA_N 50 --GA_C 0.95 --GA_M 0.02

表示求解TSP问题，使用GA算法，运行模式是显示效用值历史，地图生成方式是随机生成，GA算法参数为：--GA_N 50 --GA_C 0.95 --GA_M 0.02

## TSP问题求解应用程序
ui文件夹里是应用程序代码，使用python的pyqt模块实现。
功能：
- 应用程序可以选择GA/SA算法，可以设定二者的算法参数
- TSP地图可以选择老师提供的地图，也可以随机生成任意城市数的地图
- 界面中实时显示算法当前搜到的最优路线
- 实时显示最优路线的长度，最终显示算法用时。

## 可执行程序
由于应用程序的可执行程序过大（50 MB），我们放到了清华网盘里：
[https://cloud.tsinghua.edu.cn/f/c87ba033b2324c52ad99/](https://cloud.tsinghua.edu.cn/f/c87ba033b2324c52ad99/ "可执行程序下载链接")