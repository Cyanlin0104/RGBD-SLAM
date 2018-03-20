#-*- coding: utf-8 -*-
import slamBase
def transformPC(src,T):
    pointcloud = []
    for item in src:
        a = list(item)
        a.append(1)
        a = np.matrix(a)
        a= a.reshape((-1,1))
        temp = T * a
        temp = temp.reshape((-1,1))

