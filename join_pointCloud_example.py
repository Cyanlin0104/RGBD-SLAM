import slamBase as sb
import cv2
import numpy as np
from concurrent import futures
import time

# Load Images to List
imgPath = '/home/lin/RGBD_slam/data/shiyanshiAll_temp/'
cameraDataPath = '/home/lin/RGBD_slam/data/kinect.yml'

rgbList = []
depthList = []

for i in range(1,15):
    rgbList.append( cv2.imread(imgPath + 'rgb{}.png'.format(i)) )
    depthList.append( cv2.imread(imgPath + 'depth{}.png'.format(i),cv2.COLOR_BGR2GRAY))

camera = sb.Camera(cameraDataPath)
pnp = sb.SolvePnP(camera)

frameList = []
print ('Converting images to pointClouds')

t0 = time.time()
for i in xrange(len(rgbList)):
    frameList.append(sb.FramePC(rgbList[i], depthList[i], camera))
t1 = time.time()

print (t1-t0)

TM = []
for i in xrange(len(rgbList) - 1):
    TM_ = pnp.solveFrame(frameList[i+1], frameList[i])
    TM.append(TM_)

    PC_joined = frameList[0].pointCloud

for i in range(len(rgbList) - 1):
    T = np.identity(4)
    for l in range(i+1):
        T = np.dot(T, TM[l])
    PC_transformed = sb.transformPC(frameList[i + 1].pointCloud, TransformMatrix=T)
    PC_joined = np.concatenate((PC_joined, PC_transformed),axis=0)
    colors_joined = []
for i in xrange(len(frameList)):
    colors_joined += frameList[i].colors
    

sb.savePC(PC_joined,imgPath + '806_1.pcd',colors_joined)