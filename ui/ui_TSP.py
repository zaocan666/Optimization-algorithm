from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QDesktopWidget, QMessageBox, QPushButton, QLabel, QLineEdit, QComboBox
from PyQt5.QtGui import QPainter, QBrush, QIntValidator
from PyQt5.QtCore import Qt, QBasicTimer
# from PyQt5 import QtGui
import sys
import numpy as np
import copy
import time

from GA_algorithm import GA_optimizer

# from game_class import Solver, get_small_tri_num
# from pic_process import tooClose, arc_len, num_tooClose
# from ui_basic import Solve_frame


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui_userD = Ui_TSP()
        self.ui_userD.setParent(self)
        self.ui_userD.initUI()

        self.resize(1200, 700)
        self.center()
        self.setWindowTitle('Tangram')
        self.show()

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2,
                  (screen.height() - size.height()) / 2)

class Ui_TSP(QWidget):
    def __init__(self):
        super().__init__()

    def initUI(self):
        self.resize(1200, 700)

        self.points_list = ['BEN30-XY.txt', 'BEN50-XY.txt', 'BEN75-XY.txt']
        self.points_base_root = 'TSP_points/'

        self.Point_area = TSP_MAP(self)
        self.Point_area.read_from_file(self.points_base_root+self.points_list[0])
        self.Point_area.move(20, 50)

        points_choose_label = QLabel(self)
        points_choose_label.move(self.Point_area.x()+self.Point_area.width()+30, self.Point_area.y()+20)
        points_choose_label.setText("选择预置点：")
        self.points_choose_combo = QComboBox(self)
        self.points_choose_combo.move(points_choose_label.x()+points_choose_label.width()+30, points_choose_label.y())
        self.points_choose_combo.resize(150,self.points_choose_combo.height())
        self.points_choose_combo.addItems(self.points_list)
        self.points_choose_combo.currentIndexChanged.connect(self.points_change)

        p_num_label = QLabel(parent = self)
        p_num_label.setText("点个数：")
        p_num_label.move(points_choose_label.x(), points_choose_label.y()+80)
        self.p_num_line = QLineEdit(parent = self)
        self.p_num_line.setText("30")
        self.p_num_line.move(p_num_label.x()+p_num_label.width()+30, p_num_label.y())

        random_generate_button = QPushButton(parent=self)
        random_generate_button.setText("自动生成")
        random_generate_button.move(p_num_label.x()+80, p_num_label.y() + p_num_label.height() + 20)
        random_generate_button.pressed.connect(self.random_generate)

        self.solve_frame = Solve_frame(self, [500, 1000])
        self.solve_frame.move(p_num_label.x(), random_generate_button.y()+60)
        self.solve_frame.solve_problem_button.pressed.connect(lambda :self.solve_frame.solve_button_pressed(self.Point_area))

    def points_change(self):
        # self.solver_class.solver=None
        # self.solver_class.playing_index = -1
        # self.solver_class.timer.stop()

        current_text = self.points_choose_combo.currentText()
        self.Point_area.read_from_file(self.points_base_root+current_text)
        self.repaint()
        #print(current_text)

    def random_generate(self):
        try:
            p_num = int(self.p_num_line.text())
            assert p_num>=0
        except:
            QMessageBox.information(self, "提示", "点个数输入必须为正整数", QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)
            return

        self.Point_area.num_of_points = p_num
        self.Point_area.random_generate()
        self.repaint()

class GA_args(QWidget):
    def __init__(self, parent):
        super().__init__()

        self.setParent(parent)
        self.resize(300, 200)

        a0_label = QLabel(parent = self)
        a0_label.setText("最大迭代次数：")
        a0_label.move(0, 0)
        self.a0_line = QLineEdit(parent = self)
        self.a0_line.setText("2000")
        self.a0_line.move(a0_label.x()+a0_label.width()+30, a0_label.y())

        a1_label = QLabel(parent = self)
        a1_label.setText("种群规模：")
        a1_label.move(a0_label.x(), a0_label.y()+a0_label.height()+5)
        self.a1_line = QLineEdit(parent = self)
        self.a1_line.setText("60")
        self.a1_line.move(a1_label.x()+a1_label.width()+30, a1_label.y())

        a2_label = QLabel(parent=self)
        a2_label.setText("交叉概率：")
        a2_label.move(a1_label.x(), a1_label.y()+a1_label.height()+5)
        self.a2_line = QLineEdit(parent=self)
        self.a2_line.setText("0.95")
        self.a2_line.move(self.a1_line.x(), a2_label.y())

        a3_label = QLabel(parent=self)
        a3_label.setText("变异概率：")
        a3_label.move(a1_label.x(), a2_label.y()+a2_label.height()+5)
        self.a3_line = QLineEdit(parent=self)
        self.a3_line.setText("0.2")
        self.a3_line.move(self.a1_line.x(), a3_label.y())

        a4_label = QLabel(parent=self)
        a4_label.setText("收敛不变上限：")
        a4_label.move(a1_label.x(), a3_label.y()+a3_label.height()+5)
        self.a4_line = QLineEdit(parent=self)
        self.a4_line.setText("200")
        self.a4_line.move(self.a1_line.x(), a4_label.y())

        a5_label = QLabel(parent=self)
        a5_label.setText("上一种群保留比例：")
        a5_label.move(a1_label.x(), a4_label.y()+a4_label.height()+5)
        self.a5_line = QLineEdit(parent=self)
        self.a5_line.setText("0.2")
        self.a5_line.move(self.a1_line.x(), a5_label.y())

    def get_args(self):
        def int_arg(line, name):
            try:
                x = int(line.text())
                assert x>0
            except:
                QMessageBox.information(self, "提示", name+"必须为正整数", QMessageBox.Ok | QMessageBox.Close,
                                        QMessageBox.Close)
                return []
            return x

        def float_arg(line, name):
            try:
                x = float(line.text())
                assert x>=0
                assert x<=1
            except:
                QMessageBox.information(self, "提示", name+"必须为正整数", QMessageBox.Ok | QMessageBox.Close,
                                        QMessageBox.Close)
                return []
            return x

        iter = int_arg(self.a0_line, "最大迭代次数")
        N = int_arg(self.a1_line, "种群规模")
        C = float_arg(self.a2_line, "交叉概率")
        M = float_arg(self.a3_line, "变异概率")
        nochange_iter = int_arg(self.a4_line, "收敛不变上限")
        lastgl = float_arg(self.a3_line, "上一种群保留比例")
        
        result = [iter, N, C, M, nochange_iter, lastgl]
        if [] in result:
            return []

        return result

class SA_args(QWidget):
    def __init__(self, parent):
        super().__init__()

        self.setParent(parent)
        self.resize(300, 200)

        a0_label = QLabel(parent = self)
        a0_label.setText("最大迭代次数：")
        a0_label.move(0, 0)
        self.a0_line = QLineEdit(parent = self)
        self.a0_line.setText("5000")
        self.a0_line.move(a0_label.x()+a0_label.width()+30, a0_label.y())

        # a1_label = QLabel(parent = self)
        # a1_label.setText("种群规模：")
        # a1_label.move(a0_label.x(), a0_label.y()+a0_label.height()+5)
        # self.a1_line = QLineEdit(parent = self)
        # self.a1_line.setText("60")
        # self.a1_line.move(a1_label.x()+a1_label.width()+30, a1_label.y())

        # a2_label = QLabel(parent=self)
        # a2_label.setText("交叉概率：")
        # a2_label.move(a1_label.x(), a1_label.y()+a1_label.height()+5)
        # self.a2_line = QLineEdit(parent=self)
        # self.a2_line.setText("0.95")
        # self.a2_line.move(self.a1_line.x(), a2_label.y())

        # a3_label = QLabel(parent=self)
        # a3_label.setText("变异概率：")
        # a3_label.move(a1_label.x(), a2_label.y()+a2_label.height()+5)
        # self.a3_line = QLineEdit(parent=self)
        # self.a3_line.setText("0.2")
        # self.a3_line.move(self.a1_line.x(), a3_label.y())

        # a4_label = QLabel(parent=self)
        # a4_label.setText("收敛不变上限：")
        # a4_label.move(a1_label.x(), a3_label.y()+a3_label.height()+5)
        # self.a4_line = QLineEdit(parent=self)
        # self.a4_line.setText("500")
        # self.a4_line.move(self.a1_line.x(), a4_label.y())

        # a5_label = QLabel(parent=self)
        # a5_label.setText("上一种群保留比例：")
        # a5_label.move(a1_label.x(), a4_label.y()+a4_label.height()+5)
        # self.a5_line = QLineEdit(parent=self)
        # self.a5_line.setText("0.2")
        # self.a5_line.move(self.a1_line.x(), a5_label.y())

    def get_args(self):
        def int_arg(line, name):
            try:
                x = int(line.text())
                assert x>0
            except:
                QMessageBox.information(self, "提示", name+"必须为正整数", QMessageBox.Ok | QMessageBox.Close,
                                        QMessageBox.Close)
                return []
            return x

        def float_arg(line, name):
            try:
                x = float(line.text())
                assert x>=0
                assert x<=1
            except:
                QMessageBox.information(self, "提示", name+"必须为正整数", QMessageBox.Ok | QMessageBox.Close,
                                        QMessageBox.Close)
                return []
            return x

        iter = int_arg(self.a0_line, "最大迭代次数")
        # N = int_arg(self.a1_line, "种群规模")
        # C = float_arg(self.a2_line, "交叉概率")
        # M = float_arg(self.a3_line, "变异概率")
        # nochange_iter = int_arg(self.a4_line, "收敛不变上限")
        # lastgl = float_arg(self.a3_line, "上一种群保留比例")
        
        result = [iter]
        if [] in result:
            return []

        return result

class TSP_MAP(QWidget):
    def __init__(self, parent, num_of_points=0):
        super().__init__()
        self.resize(60 * 12, 60 * 10)
        self.setParent(parent)
        self.num_of_points = num_of_points
        self.route = []

    # 计算距离矩阵 distance_martix[i, j] = distance(points[i], points[j])
    def calculate_distance_martix(self):
        G = np.dot(self.points, self.points.T)  # [n, n]
        H = np.tile(np.diag(G), (self.points.shape[0], 1))
        distance_martix = H + H.T - 2 * G
        self.distance_martix = np.power(distance_martix, 0.5)

    def random_generate(self):
        self.points = np.random.rand(self.num_of_points, 2)*100
        self.calculate_distance_martix()
        self.get_draw_points()
        self.route = []
        
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
        self.get_draw_points()
        self.route = []

    def get_draw_points(self):
        scale1 = (self.width()-5)/self.points[:,0].max()
        scale2 = (self.height()-5)/self.points[:,1].max()
        scale = min([scale1, scale2])

        self.draw_points = np.zeros(self.points.shape)
        self.draw_points[:, 0] = self.points[:, 0] * scale
        self.draw_points[:, 1] = self.points[:, 1] * scale
        self.draw_points[:, 1] = self.height() - self.draw_points[:, 1]

    def route_distance(self, route):
        distance_sum = 0
        for i in range(self.num_of_points - 1):
            distance_sum += self.distance_martix[route[i], route[i + 1]]
        distance_sum += self.distance_martix[route[-1], route[0]]

        return distance_sum
    
    def paintEvent(self, QPaintEvent):
        qp = QPainter(self)
        self.draw_all_points(qp)
        self.draw_route(qp)

    def draw_all_points(self, qp):
        brush = QBrush(Qt.SolidPattern)
        brush.setColor(Qt.black)
        qp.setBrush(brush)

        radius = 4
        for p in self.draw_points:
            qp.drawEllipse(p[0]-radius, p[1]-radius, radius*2, radius*2)

    def draw_route(self, qp):
        if len(self.route)==0:
            return

        brush = QBrush(Qt.SolidPattern)
        brush.setColor(Qt.red)
        qp.setBrush(brush)

        for i in range(self.num_of_points - 1):
            qp.drawLine(self.draw_points[self.route[i], 0], self.draw_points[self.route[i], 1],
                            self.draw_points[self.route[i+1], 0], self.draw_points[self.route[i+1], 1])	
        qp.drawLine(self.draw_points[self.route[-1], 0], self.draw_points[self.route[-1], 1],
                        self.draw_points[self.route[0], 0], self.draw_points[self.route[0], 1])	

#将解答按钮、解答过程播放按钮和速度选择框封装成一个类，供三个标签页重复使用
class Solve_frame(QWidget):
    def __init__(self, parent, size):
        super().__init__()

        self.playing_index = -1
        self.solver = None
        self.problem_solving = False

        algorithm_label = QLabel(self)
        algorithm_label.move(0, 20)
        algorithm_label.setText("算法：")
        self.algorithm_combo = QComboBox(self)
        self.algorithm_combo.move(algorithm_label.x()+algorithm_label.width()+30, algorithm_label.y())
        self.algorithm_combo.resize(150,self.algorithm_combo.height())
        self.algorithm_combo.addItems(['GA', 'SA'])
        self.algorithm_combo.currentIndexChanged.connect(self.algorithm_change)

        self.ga_args = GA_args(self)
        self.ga_args.move(algorithm_label.x(), algorithm_label.y()+50)
        self.sa_args = SA_args(self)
        self.sa_args.move(algorithm_label.x(), algorithm_label.y()+50)
        self.sa_args.setHidden(True)

        self.setParent(parent)
        self.myparent = parent
        self.resize(*size)

        self.solve_problem_button = QPushButton(parent=self)
        self.solve_problem_button.setText("解答")
        self.solve_problem_button.move(self.ga_args.x()+80, self.ga_args.y()+self.ga_args.height()+30)

        self.solve_test = QLabel(parent=self) #解答过程中的信息显示
        self.solve_test.setText("")
        self.solve_test.resize(250, self.solve_test.height())
        self.solve_test.move(20, self.solve_problem_button.y() + self.solve_problem_button.height() + 10)
        # self.solve_test.setHidden(True)

        speed_choose_label = QLabel(self)
        speed_choose_label.move(20, self.solve_test.y() + self.solve_test.height() + 40)
        speed_choose_label.setText("解答速度：")
        self.play_speed_combo = QComboBox(self)
        self.play_speed_combo.move(speed_choose_label.x() + speed_choose_label.width() + 30,
                                   speed_choose_label.y())
        self.play_speed_combo.addItems(["高速", "中速", "慢速"])

        self.stop_button = QPushButton(parent=self)
        self.stop_button.setText("终止")
        self.stop_button.move(self.solve_problem_button.x(), self.play_speed_combo.y()+self.play_speed_combo.height()+20)
        self.stop_button.setHidden(True)
        self.stop_button.pressed.connect(self.stop_timer)

        self.timer = QBasicTimer()

    def algorithm_change(self):
        if self.algorithm_combo.currentText()=='GA':
            self.ga_args.setHidden(False)
            self.sa_args.setHidden(True)
        else:
            self.ga_args.setHidden(True)
            self.sa_args.setHidden(False)

    def solve_button_pressed(self, tsp_map):
        if self.problem_solving:
            return
        if tsp_map.num_of_points<=0:
            QMessageBox.information(self, "警告", "地图中无点", QMessageBox.Ok)
            return

        self.tsp_map = tsp_map

        if self.algorithm_combo.currentText()=='GA':
            args = self.ga_args.get_args()
            if args==[]:
                return
            self.optimizer = GA_optimizer(tsp_map, *args[1:], history_convert=lambda x: -x)
        else:
            args = self.sa_args.get_args()
            if args==[]:
                return

        self.max_iteration = args[0]

        self.problem_solving = True
        self.iter_num = 0
        self.stop_button.setHidden(False)
        self.timer.stop()
        self.repaint()

        self.start_time = time.time()

        speed_text = self.play_speed_combo.currentText()
        self.playing_index = 0
        if speed_text == "高速":
            self.timer.start(0, self)
        elif speed_text == "中速":
            self.timer.start(100, self)
        else:
            self.timer.start(500, self)

    def timerEvent(self, event):
        if event.timerId() == self.timer.timerId():
            route, distance = self.optimizer.step()

            if self.iter_num >= self.max_iteration or (not route):
                end_time = time.time()
                QMessageBox.information(self, "提示", "完成解答，用时：%.1f s" % (end_time - self.start_time),
                                QMessageBox.Ok | QMessageBox.Close, QMessageBox.Close)
                
                # self.solve_test.setHidden(True)
                self.end()

            if route:
                self.parent().Point_area.route = route
                self.parent().repaint()
                self.solve_test.setText("回合数:%d 适配值:%.2f"%(self.iter_num, distance))

            self.iter_num += 1

        else:
            super(Solve_frame, self).timerEvent(event)

    def stop_timer(self):
        self.end()

    def end(self):
        self.iter_num=0
        self.problem_solving = False
        self.stop_button.setHidden(True)
        self.repaint()
        self.timer.stop()

if __name__ == "__main__":
    app = QApplication([])
    ui = MainWindow()
    sys.exit(app.exec_())