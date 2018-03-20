#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File :
	
	slamBase.py

Description:
		
	A python lib that provide a fundamental Architecture for RGBD SLAM 

Author:
	created by CyanLin
	
	2017 / 12 / 07 / 21:41 Thursday	
	

"""


import cv2
import pcl
import readyaml
import numpy as np

testMode = True

class Frame(object):
    
    def __init__(self,rgbImg,depthImg):
        self.rgb = rgbImg
        self.depth = depthImg

class FramePC(Frame):
    
    def __init__(self,rgbImg,depthImg,camera):
        super(FramePC,self).__init__(rgbImg,depthImg)

        self.pointCloud , self.colors = imageToPointCloud(self.rgb,self.depth, camera,
                                                          cloudFileName = None)
        self.keyPoints = None
            
class CameraIntrinsicParameters(object):
    
    def __init__(self,cx,cy,fx,fy,scale):
        super(CameraIntrinsicParameters,self).__init__()
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy
        self.scale = scale
        
class Camera(CameraIntrinsicParameters):
    
    def __init__(self,calibrationFilePath):
        self.IntrinsicData, self.distCoeff = readyaml.parseYamlFile(calibrationFilePath)
        super(Camera,self).__init__(self.IntrinsicData[0][2],self.IntrinsicData[1][2],self.IntrinsicData[0][0],
            self.IntrinsicData[1][1],1000.0)
         
        
class SolvePnP(object):  
    
    def __init__(self,camera):

    #camera : A instance of class Camera that describe camera intrinsic and distortion parameters
    
    # initialize Camera relatives
        super(SolvePnP,self).__init__()
    
        self.camera = camera
        self.CameraIntrinsicData = np.array([[camera.fx,   0.,     camera.cx],
                                             [0.,        camera.fy,camera.cy],
                                             [0.,          0.,            1.]])  
        self.goodMatches = None    
    
    def solveFrame(self,frame1,frame2):
        # if there is not a large number of frame , 
        # save keyPoints to class FramePC
        self.goodMatches, frame1.keyPoints, frame2.keyPoints = featureMatch(frame1.rgb, frame2.rgb) 
        
        global testMode
        if(testMode):  
            print("good matches = " + str(len(self.goodMatches)) )


        
        pts_3d_query = []  # 3D points from source image
        pts_2d_train = []  # 2D points from target image
        for goodMatch in self.goodMatches:
            p = frame1.keyPoints[goodMatch.queryIdx].pt
            d = frame1.depth[int(p[1])][int(p[0])]
           
            if (d == 0):
                continue
            pts_2d_train.append(frame2.keyPoints[goodMatch.trainIdx].pt)
            point3D_query = point2DTO3D(p[0],p[1],d,self.camera) 
            pts_3d_query.append(point3D_query)
        
        
        
        IntrinsicData = self.camera.IntrinsicData    
        distCoeffs = self.camera.distCoeff
        print 'solving PnP'
        _, rvec, tvec, inliers = cv2.solvePnPRansac(np.array(pts_3d_query),np.array(pts_2d_train),
                                                 np.matrix(IntrinsicData),distCoeffs,
                                                 useExtrinsicGuess = False, iterationsCount = 100,
                                                 reprojectionError = 1.76)
             
        lastRow = np.array([[0.,0.,0.,1.]])
        r = np.matrix(rvec)
        t = np.matrix(tvec)
        # convert a rotation vector(represent in opencv) to rotation matrix
        dst, _ = cv2.Rodrigues(r)
        temp = np.concatenate((dst,t),axis = 1)
       
        TransformMatrix = np.concatenate((temp,lastRow),axis = 0)

        return np.array(TransformMatrix)     
        
        """
      cv2.solvePnPRansac(objectPoints, imagePoints,
      					cameraMatrix, distCoeffs[, rvec[, tvec[, useExtrinsicGuess
     					 [, iterationsCount[, reprojectionError[, minInliersCount[, inliers[, flags]]]]]]]]) 
   	   → retvel , rvec, tvec, inliers

        """    

        


def point2DTO3D(u,v,d,camera):
    """
    A Essential function to process point form 2D to 3D which implemented and based on "pinhole model"  
    Args: 
    	camera - An Instance of Class : camera intrinsic parameters - cx,cy,fx,fy
        u , v - : 2D coordinates - (u,v)
        d - : Depth in this coordinate - d

    Returns: 
    	point3D - A numpy (1,3)array : 3D coordinates (x,y,z)  
    """
    z = float(d) / camera.scale
    x = (u - camera.cx) * z / camera.fx
    y = (v - camera.cy) * z / camera.fy
    point3D = [x,y,z]

    return point3D


def imageToPointCloud(rgb,depth,camera,cloudFileName = None):

    """
    #if entered a cloudFilename, convert image to a pointCloud file,
    #else, return a numpy array contains pointcloud 
    
    """
    pointCloud = []
    colors = []
    for v in range(0, len(depth)):
        for u in range(0, len(depth[0])):
            d = depth[v][u]
            if (d == 0):
                pass
            else:
                point3D = point2DTO3D(u,v,d,camera)
                pointCloud.append(point3D)
                b = rgb[v][u][0]
                g = rgb[v][u][1]
                r = rgb[v][u][2]
                color = (r << 16) | (g << 8) | b
                colors.append(int(color))
                
    pointcloud = np.array(pointCloud, dtype = np.float32)
    if cloudFileName is not None:
        savePC(pointcloud,cloudFileName,colors)
    else:
        return pointcloud , colors
    
    
def featureMatch(rgb1,rgb2,method = cv2.xfeatures2d.SIFT_create()):
    # 1. compute and detect feature
    kp1, des1 = method.detectAndCompute(rgb1,None)
    kp2, des2 = method.detectAndCompute(rgb2,None)
    
    # 2. match
    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm = FLANN_INDEX_KDTREE,trees = 5)
    search_params = dict(checks = 50)
    flann_matcher = cv2.FlannBasedMatcher(indexParams,search_params)
    matches = flann_matcher.match(des1,des2)

    # 3. find good matches
    goodMatches = []
    minDis = matches[0].distance 

    for i in range(len(matches)):
    	if matches[i].distance < minDis:
    		minDis = matches[i].distance

    for match in matches:
        if(match.distance < 3 * minDis):
            goodMatches.append(match)
    
    global testMode
    if (testMode):
        imgMatches = None
        imgMatches = cv2.drawMatches(rgb1,kp1,rgb2,kp2,goodMatches,imgMatches)
        cv2.imshow('testMode',imgMatches)
        if(cv2.waitKey(0)):
            cv2.destroyAllWindows()
                    
    return goodMatches, kp1, kp2
        

def transformPC(points3D,rvec = None,tvec = None,TransformMatrix = None):
    
    if TransformMatrix is None:
        lastRow = np.array([[0.,0.,0.,1.]])
        r = np.matrix(rvec)
        t = np.matrix(tvec)
        # convert a rotation vector(represent in opencv) to rotation matrix
        dst, _ = cv2.Rodrigues(r)
        temp = np.concatenate((dst,t),axis = 1)
       
        TM = np.concatenate((temp,lastRow),axis = 0)
        TM = np.array(TM)
    else:
        TM = TransformMatrix 
    # Transform Matrix T(4x4) 
  
    pointcloud = []    
    for point3D in points3D:
        
        point3D = np.array(point3D, dtype = np.float32)# point3D(3,)
        point3D = np.concatenate((point3D, np.array([1.])), axis = 0) #point3D(4,)
      
        point3D_t = np.dot(TM,point3D) #point3D_t(4,)
        point3D_t = np.delete(point3D_t,-1,axis = 0)
    
        pointcloud.append(point3D_t)   
    global testMode
    if(testMode): print TM    
    
    pointcloud = np.array(pointcloud,dtype = np.float32)
    return pointcloud

 
def addColorToPCDFile(filename,colors):
    with open(filename,'rb') as f:
        lines = f.readlines()
        lines[2] = lines[2].split('\n')[0] + ' rgb\n'
        lines[3] = lines[3].split('\n')[0] + ' 4\n'
        lines[4] = lines[4].split('\n')[0] + ' I\n'
        lines[5] = lines[5].split('\n')[0] + ' 1\n'
        for i in range(11,len(colors) + 11):
            lines[i] = lines[i].split('\n')[0] + ' ' + str(colors[i - 11]) + '\n'
        with open(filename,'wb') as fw:
            fw.writelines(lines)

def savePC(pointcloud,fileName,colors = None):
    cloud = pcl.PointCloud()
    cloud.from_array(pointcloud)
    pcl.save(cloud,fileName,format='pcd')
    if(colors is not None): 
    	addColorToPCDFile(fileName,colors)
    print 'PCD file Saved'
    
        
        
        
    

        
if __name__ == '__main__' :
    
    
    imgPath = '/home/lin/internal-save/'
    calibrationFilePath = '/home/lin/RGBD_slam/data/camera.yml'
    saveFileName = '/home/lin/RGBD_slam/data/bed.pcd'
    
    
    
    rgb1 = cv2.imread(imgPath + 'rgb3.png')
    rgb2 = cv2.imread(imgPath + 'rgb4.png')
    depth1 = cv2.imread(imgPath + 'depth3.png',cv2.COLOR_BGR2GRAY)
    depth2 = cv2.imread(imgPath + 'depth4.png',cv2.COLOR_BGR2GRAY)
    camera = Camera(calibrationFilePath)


#    test SolvePnP
    
    frame1 = FramePC(rgb1,depth1,camera)
    frame2 = FramePC(rgb2,depth2,camera)
    
    pnp = SolvePnP(camera)
    
    TM = pnp.solveFrame(frame1,frame2)
    
    PC1 = transformPC(frame1.pointCloud,TransformMatrix=TM)
    PC2 = frame2.pointCloud
    print 'shape of PC1 ' + str(PC1.shape)
    print 'shape of PC2'  + str(PC2.shape)
    
    PC_joined = np.concatenate((PC1,PC2),axis = 0)
    colors_joined = frame1.colors + frame2.colors
    
    savePC(PC_joined,saveFileName,colors_joined)
       
    
#2017年 11月 21日 星期二 06:23:43 CST
