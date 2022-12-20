# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea

class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [1]  # 初始化 maxormins （目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 18  # 初始化Dim（决策变量维数）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [3, 0, 0, 0, 0, 0, 3, 3,
              0, 0,
              0, 0, 0, 0, 0, 0, 0, 0]  # 决策变量下界 ,前八个是op所在节点的位置
        # 中间八个是op所使用的broker的位置
        # 最后两个每个broker所在节点的位置
        ub = [3, 3, 3, 3, 3, 3, 3, 3,
              3, 3,
              1, 1, 1, 1, 1, 1, 1, 1]  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)


        # 添加属性来存储链路
        self.link = np.array([[0, 1, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1, 1],
                              [0, 0, 0, 0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0]])

    def aimFunc(self, pop):
        # 目标函数
        x = pop.Phen.astype(int)  # 得到决策变量矩阵

        node_limit = []
        for i in range(0, len(x)):  # 各节点资源的限制
            node_limit.append([20, 20, 10000, 30,
                               10, 10, 6000, 20,
                               10, 10, 6000, 20,
                               4, 4, 3000, 6])

        X = np.hstack([x, node_limit]).astype(int)

        Objv = []
        for i in range(X.shape[0]):  # 遍历每一个种群
            sum = 0
            for j in range(0, 8):  # 把每个节点上operator的资源消耗除去
                sub = 0
                pub = 0
                pub1 = 0
                sub1 = 0
                flag = 0
                for k in range(0, 8):
                    a = int(self.link[j][k])  # j->k判断是否有连接
                    b = int(self.link[k][j])  # k->j判断是否有连接
                    if a > 0:
                        if X[i][j] == X[i][k] & X[i][j] == X[i][X[i][j + 10] + 8]: # 判断是否为节点内部传输
                            pub1 = pub1
                        else:
                            pub1 += 1
                        if X[i][j] != X[i][k] & X[i][j] != X[i][X[i][j + 10] + 8]:
                            flag += 1

                        pub += 1
                    if b > 0:
                        if X[i][j] == X[i][k] & X[i][j] == X[i][X[i][k + 10] + 8]:
                            sub1 = sub1
                        else:
                            sub1 += 1
                        sub += 1
                c = int(X[i][j])  # operator j所在的节点编号
                d = int(X[i][X[i][j + 10] + 8])  # op j所在的broker所在的节点编号

                X[i][c * 4 + 18] -= pub1  # 节点c 的pub消耗
                X[i][c * 4 + 19] -= sub1  # 节点c 的sub消耗
                X[i][c * 4 + 20] -= 1000 # 节点c 的cpu消耗
                X[i][c * 4 + 21] -= 1  # 节点c 的内存消耗
                if c == d:  # 若op j所在的节点和op j发送的broker所在的节点相同，broker的消耗
                    X[i][c * 4 + 20] -= 300 * pub  # 节点c 的cpu消耗
                    X[i][c * 4 + 21] -= 0.5 * pub  # 节点c 的内存消耗
                else:  # 若op j所在的节点和op j发送的broker所在的节点不同，broker的消耗
                    X[i][d * 4 + 20] -= 300 * pub  # 节点d 的cpu消耗
                    X[i][d * 4 + 21] -= 0.5 * pub  # 节点d 的内存消耗
                    X[i][d * 4 + 19] -= flag  # 节点d 的sub消耗
                    X[i][d * 4 + 18] -= flag  # 节点d 的pub消耗


            for j in range(0, 8):
                time_pub = 0
                time_sub = 0
                pub = 0
                sub1 = 0
                sub2 = 0
                sub3 = 0
                for k in range(0, 8):
                    a = int(self.link[j][k])  # j->k判断是否有连接
                    if a == 1:# 节点 j->k有发出
                        if X[i][k] == 0 & X[i][k] != X[i][j]:
                            sub1 += 1
                        elif (X[i][k] == 1 or X[i][k] == 2) & X[i][k] != X[i][j]:
                            sub2 += 1
                        elif X[i][k] == 3 & X[i][k] != X[i][j]:
                            sub3 += 1
                pub += sub1 + sub2 + sub3
                        # print("X[i][k]",X[i][k],"sub1",sub1,"sub1",sub2,"sub1",sub3)
                # print("pub",pub)
                # print("sub1",sub1)
                # print("sub2", sub2)
                # print("sub3", sub3)
                # op->broker传输消耗时间
                op_node = X[i][j]  # op j 所在的节点编号
                bro_node = X[i][X[i][j + 10] + 8]  # op j所在的broker所在的节点编号
                if op_node == bro_node:  # op与bro所在的节点相同
                    time_pub = 0
                elif (op_node == 0 & bro_node == 1) or (op_node == 1 & bro_node == 0):
                    time_pub = pub * 30
                elif (op_node == 0 & bro_node == 2) or (op_node == 2 & bro_node == 0):
                    time_pub = pub * 30
                elif (op_node == 0 & bro_node == 3) or (op_node == 3 & bro_node == 0):
                    time_pub = pub * 50
                elif (op_node == 1 & bro_node == 2) or (op_node == 2 & bro_node == 1):
                    time_pub = pub * 25
                elif (op_node == 1 & bro_node == 3) or (op_node == 3 & bro_node == 1):
                    time_pub = pub * 20
                elif (op_node == 3 & bro_node == 2) or (op_node == 2 & bro_node == 3):
                    time_pub = pub * 20
                # broker->op传输消耗时间及broker处理时间
                if bro_node == 0:  # bro在云端
                    time_sub = sub1 * 40 + sub2 * 30 + sub3 * 50 + pub * 1
                elif bro_node == 1:  # bro在雾端
                    time_sub = sub1 * 30 + sub2 * 25 + sub3 * 20 + pub * 15
                elif bro_node == 2:  # bro在雾端
                    time_sub = sub1 * 30 + sub2 * 25 + sub3 * 20 + pub * 15
                elif bro_node == 3:  # bro在边缘端
                    time_sub = sub1 * 50 + sub2 * 20 + sub3 * 10 + pub * 30
                # print("time_sub",time_sub)
                # print("time_pub",time_pub)
                sum += time_sub + time_pub
                # print("sum",sum)
            print(sum)
            print(X[i])
            Objv.append(sum)
        pop.ObjV = np.array([Objv]).T  # 把求得的目标函数值赋值给种群pop的ObjV

        x1 = X[:, [18]]
        x2 = X[:, [19]]
        x3 = X[:, [20]]
        x4 = X[:, [21]]
        x5 = X[:, [22]]
        x6 = X[:, [23]]
        x7 = X[:, [24]]
        x8 = X[:, [25]]
        x9 = X[:, [26]]
        x10 = X[:, [27]]
        x11 = X[:, [28]]
        x12 = X[:, [29]]
        x13 = X[:, [30]]
        x14 = X[:, [31]]
        x15 = X[:, [32]]
        x16 = X[:, [33]]
        # 采用可行性法则处理约束

        exIdx1 = np.where(x1 < 0)[0]
        exIdx2 = np.where(x2 < 0)[0]
        exIdx3 = np.where(x3 < 0)[0]
        exIdx4 = np.where(x4 < 0)[0]
        exIdx5 = np.where(x5 < 0)[0]
        exIdx6 = np.where(x6 < 0)[0]
        exIdx7 = np.where(x7 < 0)[0]
        exIdx8 = np.where(x8 < 0)[0]
        exIdx9 = np.where(x9 < 0)[0]
        exIdx10 = np.where(x10 < 0)[0]
        exIdx11 = np.where(x11 < 0)[0]
        exIdx12 = np.where(x12 < 0)[0]
        exIdx13 = np.where(x13 < 0)[0]
        exIdx14 = np.where(x14 < 0)[0]
        exIdx15 = np.where(x15 < 0)[0]
        exIdx16 = np.where(x16 < 0)[0]
        exIdx = np.unique(np.hstack([exIdx1, exIdx2, exIdx3, exIdx4, exIdx5, exIdx6, exIdx7, exIdx8,
                                     exIdx9, exIdx10, exIdx11, exIdx12, exIdx13, exIdx14, exIdx15, exIdx16]))
        pop.CV = np.zeros((pop.sizes, 1))
        pop.CV[exIdx] = 1  # 把求得的违反约束程度矩阵赋值给种群pop的CV
        # pop.CV = np.hstack([x1,
        #                     x2,
        #                     x3,
        #                     x4,
        #                     x5,
        #                     x6,
        #                     x7,
        #                     x8,
        #                     x9,
        #                     x10,
        #                     x11,
        #                     x12,
        #                     x13,
        #                     x14,
        #                     x15,
        #                     x16])
        #
        # print('pop.CV')
        # print(pop.CV)
