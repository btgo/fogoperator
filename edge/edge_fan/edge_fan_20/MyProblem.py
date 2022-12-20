# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea

class MyProblem(ea.Problem):  # 继承Problem父类
        def __init__(self):
            name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
            M = 1  # 初始化M（目标维数）
            maxormins = [1]  # 初始化 maxormins （目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
            Dim = 46  # 初始化Dim（决策变量维数）
            varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
            lb = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 决策变量下界 ,前八个是op所在节点的位置
            # 中间八个是op所使用的broker的位置
            # 最后两个每个broker所在节点的位置
            ub = [23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
                  23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
                  23, 23, 23, 23, 23, 23,
                  5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                  5, 5, 5, 5, 5, 5, 5, 5, 5, 5]  # 决策变量上界
            lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
            ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
            # 调用父类构造方法完成实例化
            ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

            # 添加属性来存储链路
            self.link = np.array([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                                  ])

        def aimFunc(self, pop):
            # 目标函数
            x = pop.Phen.astype(int)  # 得到决策变量矩阵

            node_limit = []
            for i in range(0, len(x)):  # 各节点资源的限制
                node_limit.append([10, 10, 5000, 6,
                                   10, 10, 3000, 6,
                                   10, 10, 5000, 6,
                                   10, 10, 3000, 6,
                                   10, 10, 5000, 6,
                                   10, 10, 3000, 6,
                                   10, 10, 5000, 6,
                                   10, 10, 3000, 6,

                                   10, 10, 5000, 6,
                                   10, 10, 3000, 6,
                                   10, 10, 5000, 6,
                                   10, 10, 3000, 6,
                                   10, 10, 5000, 6,
                                   10, 10, 3000, 6,
                                   10, 10, 5000, 6,
                                   10, 10, 3000, 6,

                                   10, 10, 5000, 6,
                                   10, 10, 3000, 6,
                                   10, 10, 5000, 6,
                                   10, 10, 3000, 6,
                                   10, 10, 5000, 6,
                                   10, 10, 3000, 6,
                                   10, 10, 5000, 6,
                                   10, 10, 3000, 6,
                                   ])

            X = np.hstack([x, node_limit]).astype(int)
            # print("X", X)

            Objv = []
            for i in range(X.shape[0]):  # 遍历每一个种群
                sum = 0
                for j in range(0, 20):  # 把每个节点上operator的资源消耗除去
                    sub = 0
                    pub = 0
                    pub1 = 0
                    sub1 = 0
                    flag = 0
                    for k in range(0, 20):
                        a = int(self.link[j][k])  # j->k判断是否有连接
                        b = int(self.link[k][j])  # k->j判断是否有连接
                        if a > 0:
                            if X[i][j] == X[i][k] & X[i][j] == X[i][X[i][j + 26] + 20]:  # 判断是否为节点内部传输
                                pub1 = pub1
                            else:
                                pub1 += 1
                            if X[i][j] != X[i][k] & X[i][j] != X[i][X[i][j + 26] + 20]:
                                flag += 1

                            pub += 1
                        if b > 0:
                            if X[i][j] == X[i][k] & X[i][j] == X[i][X[i][k + 26] + 20]:
                                sub1 = sub1
                            else:
                                sub1 += 1
                            sub += 1
                    c = int(X[i][j])  # operator j所在的节点编号
                    d = int(X[i][X[i][j + 26] + 20])  # op j所在的broker所在的节点编号

                    X[i][c * 4 + 46] -= pub1  # 节点c 的pub消耗
                    X[i][c * 4 + 47] -= sub1  # 节点c 的sub消耗
                    X[i][c * 4 + 48] -= 1000  # 节点c 的cpu消耗
                    X[i][c * 4 + 49] -= 1  # 节点c 的内存消耗
                    if c == d:  # 若op j所在的节点和op j发送的broker所在的节点相同，broker的消耗
                        X[i][c * 4 + 48] -= 300 * pub  # 节点c 的cpu消耗
                        X[i][c * 4 + 49] -= 0.5 * pub  # 节点c 的内存消耗
                    else:  # 若op j所在的节点和op j发送的broker所在的节点不同，broker的消耗
                        X[i][d * 4 + 48] -= 300 * pub  # 节点d 的cpu消耗
                        X[i][d * 4 + 49] -= 0.5 * pub  # 节点d 的内存消耗
                        X[i][d * 4 + 47] -= flag  # 节点d 的sub消耗
                        X[i][d * 4 + 46] -= flag  # 节点d 的pub消耗

                for j in range(0, 20):
                    time_pub = 0
                    time_sub = 0
                    pub = 0
                    sub = 0
                    for k in range(0, 20):
                        a = int(self.link[j][k])  # j->k判断是否有连接
                        if a == 1:  # 节点 j->k有发出
                            if X[i][j] != X[i][X[i][j + 26] + 20]:
                                pub += 1
                            if X[i][k] != X[i][X[i][j + 26] + 20]:
                                sub += 1

                    # print("pub", pub)
                    # print("sub", sub)

                    # op->broker传输消耗时间
                    op_node = X[i][j]  # op j 所在的节点编号
                    bro_node = X[i][X[i][j + 26] + 20]  # op j所在的broker所在的节点编号
                    if op_node == bro_node:  # op与bro所在的节点相同
                        time_pub = 0
                    else:
                        time_pub = pub * 10

                    time_sub = sub * 10 + sub * 30

                    # print("time_sub", time_sub)
                    # print("time_pub", time_pub)
                    sum += time_sub + time_pub
                    # print("sum",sum)
                # print(sum)
                # print(X[i])
                Objv.append(sum)
            pop.ObjV = np.array([Objv]).T  # 把求得的目标函数值赋值给种群pop的ObjV

            x1 = X[:, [46]]
            x2 = X[:, [47]]
            x3 = X[:, [48]]
            x4 = X[:, [49]]
            x5 = X[:, [50]]
            x6 = X[:, [51]]
            x7 = X[:, [52]]
            x8 = X[:, [53]]
            x9 = X[:, [54]]
            x10 = X[:, [55]]
            x11 = X[:, [56]]
            x12 = X[:, [57]]
            x13 = X[:, [58]]
            x14 = X[:, [59]]
            x15 = X[:, [60]]
            x16 = X[:, [61]]

            x21 = X[:, [62]]
            x22 = X[:, [63]]
            x23 = X[:, [64]]
            x24 = X[:, [65]]
            x25 = X[:, [66]]
            x26 = X[:, [67]]
            x27 = X[:, [68]]
            x28 = X[:, [69]]
            x29 = X[:, [70]]
            x30 = X[:, [71]]
            x31 = X[:, [72]]
            x32 = X[:, [73]]
            x33 = X[:, [74]]
            x34 = X[:, [75]]
            x35 = X[:, [76]]
            x36 = X[:, [77]]

            x41 = X[:, [78]]
            x42 = X[:, [79]]
            x43 = X[:, [80]]
            x44 = X[:, [81]]
            x45 = X[:, [82]]
            x46 = X[:, [83]]
            x47 = X[:, [84]]
            x48 = X[:, [85]]
            x49 = X[:, [86]]
            x50 = X[:, [87]]
            x51 = X[:, [88]]
            x52 = X[:, [89]]
            x53 = X[:, [90]]
            x54 = X[:, [91]]
            x55 = X[:, [92]]
            x56 = X[:, [93]]

            x61 = X[:, [94]]
            x62 = X[:, [95]]
            x63 = X[:, [96]]
            x64 = X[:, [97]]
            x65 = X[:, [98]]
            x66 = X[:, [99]]
            x67 = X[:, [100]]
            x68 = X[:, [101]]
            x69 = X[:, [102]]
            x70 = X[:, [103]]
            x71 = X[:, [104]]
            x72 = X[:, [105]]
            x73 = X[:, [106]]
            x74 = X[:, [107]]
            x75 = X[:, [108]]
            x76 = X[:, [109]]
            # 采用可行性法则处理约束
            x123 = X[:, [110]]
            x124 = X[:, [111]]
            x125 = X[:, [112]]
            x126 = X[:, [113]]
            x127 = X[:, [114]]
            x128 = X[:, [115]]
            x129 = X[:, [116]]
            x130 = X[:, [117]]
            x131 = X[:, [118]]
            x132 = X[:, [119]]
            x133 = X[:, [120]]
            x134 = X[:, [121]]
            x135 = X[:, [122]]
            x136 = X[:, [123]]
            x141 = X[:, [124]]
            x142 = X[:, [125]]

            x143 = X[:, [126]]
            x144 = X[:, [127]]
            x145 = X[:, [128]]
            x146 = X[:, [129]]
            x147 = X[:, [130]]
            x148 = X[:, [131]]
            x149 = X[:, [132]]
            x150 = X[:, [133]]
            x151 = X[:, [134]]
            x152 = X[:, [135]]
            x153 = X[:, [136]]
            x154 = X[:, [137]]
            x155 = X[:, [138]]
            x156 = X[:, [139]]

            x161 = X[:, [140]]
            x162 = X[:, [141]]
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

            exIdx21 = np.where(x21 < 0)[0]
            exIdx22 = np.where(x22 < 0)[0]
            exIdx23 = np.where(x23 < 0)[0]
            exIdx24 = np.where(x24 < 0)[0]
            exIdx25 = np.where(x25 < 0)[0]
            exIdx26 = np.where(x26 < 0)[0]
            exIdx27 = np.where(x27 < 0)[0]
            exIdx28 = np.where(x28 < 0)[0]
            exIdx29 = np.where(x29 < 0)[0]
            exIdx30 = np.where(x30 < 0)[0]
            exIdx31 = np.where(x31 < 0)[0]
            exIdx32 = np.where(x32 < 0)[0]
            exIdx33 = np.where(x33 < 0)[0]
            exIdx34 = np.where(x34 < 0)[0]
            exIdx35 = np.where(x35 < 0)[0]
            exIdx36 = np.where(x36 < 0)[0]

            exIdx41 = np.where(x41 < 0)[0]
            exIdx42 = np.where(x42 < 0)[0]
            exIdx43 = np.where(x43 < 0)[0]
            exIdx44 = np.where(x44 < 0)[0]
            exIdx45 = np.where(x45 < 0)[0]
            exIdx46 = np.where(x46 < 0)[0]
            exIdx47 = np.where(x47 < 0)[0]
            exIdx48 = np.where(x48 < 0)[0]
            exIdx49 = np.where(x49 < 0)[0]
            exIdx50 = np.where(x50 < 0)[0]
            exIdx51 = np.where(x51 < 0)[0]
            exIdx52 = np.where(x52 < 0)[0]
            exIdx53 = np.where(x53 < 0)[0]
            exIdx54 = np.where(x54 < 0)[0]
            exIdx55 = np.where(x55 < 0)[0]
            exIdx56 = np.where(x56 < 0)[0]

            exIdx61 = np.where(x61 < 0)[0]
            exIdx62 = np.where(x62 < 0)[0]
            exIdx63 = np.where(x63 < 0)[0]
            exIdx64 = np.where(x64 < 0)[0]
            exIdx65 = np.where(x65 < 0)[0]
            exIdx66 = np.where(x66 < 0)[0]
            exIdx67 = np.where(x67 < 0)[0]
            exIdx68 = np.where(x68 < 0)[0]
            exIdx69 = np.where(x69 < 0)[0]
            exIdx70 = np.where(x70 < 0)[0]
            exIdx71 = np.where(x71 < 0)[0]
            exIdx72 = np.where(x72 < 0)[0]
            exIdx73 = np.where(x73 < 0)[0]
            exIdx74 = np.where(x74 < 0)[0]
            exIdx75 = np.where(x75 < 0)[0]
            exIdx76 = np.where(x76 < 0)[0]

            exIdx123 = np.where(x123 < 0)[0]
            exIdx124 = np.where(x124 < 0)[0]
            exIdx125 = np.where(x125 < 0)[0]
            exIdx126 = np.where(x126 < 0)[0]
            exIdx127 = np.where(x127 < 0)[0]
            exIdx128 = np.where(x128 < 0)[0]
            exIdx129 = np.where(x129 < 0)[0]
            exIdx130 = np.where(x130 < 0)[0]
            exIdx131 = np.where(x131 < 0)[0]
            exIdx132 = np.where(x132 < 0)[0]
            exIdx133 = np.where(x133 < 0)[0]
            exIdx134 = np.where(x134 < 0)[0]
            exIdx135 = np.where(x135 < 0)[0]
            exIdx136 = np.where(x136 < 0)[0]

            exIdx141 = np.where(x141 < 0)[0]
            exIdx142 = np.where(x142 < 0)[0]

            exIdx143 = np.where(x143 < 0)[0]
            exIdx144 = np.where(x144 < 0)[0]
            exIdx145 = np.where(x145 < 0)[0]
            exIdx146 = np.where(x146 < 0)[0]
            exIdx147 = np.where(x147 < 0)[0]
            exIdx148 = np.where(x148 < 0)[0]
            exIdx149 = np.where(x149 < 0)[0]
            exIdx150 = np.where(x150 < 0)[0]
            exIdx151 = np.where(x151 < 0)[0]
            exIdx152 = np.where(x152 < 0)[0]
            exIdx153 = np.where(x153 < 0)[0]
            exIdx154 = np.where(x154 < 0)[0]
            exIdx155 = np.where(x155 < 0)[0]
            exIdx156 = np.where(x156 < 0)[0]

            exIdx161 = np.where(x161 < 0)[0]
            exIdx162 = np.where(x162 < 0)[0]




            exIdx = np.unique(np.hstack([exIdx1, exIdx2, exIdx3, exIdx4, exIdx5, exIdx6, exIdx7, exIdx8,
                                         exIdx9, exIdx10, exIdx11, exIdx12, exIdx13, exIdx14, exIdx15, exIdx16,
                                         exIdx21, exIdx22, exIdx23, exIdx24, exIdx25, exIdx26, exIdx27, exIdx28,
                                         exIdx29, exIdx30, exIdx31, exIdx32, exIdx33, exIdx34, exIdx35, exIdx36,
                                         exIdx41, exIdx42, exIdx43, exIdx44, exIdx45, exIdx46, exIdx47, exIdx48,
                                         exIdx49, exIdx50, exIdx51, exIdx52, exIdx53, exIdx54, exIdx55, exIdx56,
                                         exIdx61, exIdx62, exIdx63, exIdx64, exIdx65, exIdx66, exIdx67, exIdx68,
                                         exIdx69, exIdx70, exIdx71, exIdx72, exIdx73, exIdx74, exIdx75, exIdx76,
                                         exIdx123, exIdx124, exIdx125, exIdx126, exIdx127, exIdx128,
                                         exIdx129, exIdx130, exIdx131, exIdx132, exIdx133, exIdx134, exIdx135, exIdx136,
                                         exIdx141, exIdx142,exIdx143, exIdx144, exIdx145, exIdx146, exIdx147, exIdx148,
                                         exIdx149, exIdx150, exIdx151, exIdx152, exIdx153, exIdx154, exIdx155, exIdx156,
                                         exIdx161, exIdx162,
                                         ]))
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