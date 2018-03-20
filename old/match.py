#!usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from slamBase import point2DTO3D, CameraIntrinsicParameters
import readyaml
def readImgFiles(RGBFilenames, DepthFilenames , paras):
    rgbs = []
    depths = []
    for i in range(0,len(RGBFilenames)):
        rgbs.append(cv2.imread(RGBFilenames[i]))
    for i in range(0,len(DepthFilenames)):
        depth = cv2.imread(DepthFilenames[i],paras)
        depths.append(depth)
    return rgbs,depths

def computeMatches(rgb1,rgb2,depth1,depth2,CameraIntrinsicData,distCoeffs,camera):
    sift = cv2.xfeatures2d.SIFT_create() #same way as SURF 
    kp1, des1 = sift.detectAndCompute(rgb1,None)
    kp2, des2 = sift.detectAndCompute(rgb2,None)
    print("Key points of two images: " + str(len(kp1)) + ", " + str(len(kp2)))
    imgShow = None
    imgShow = cv2.drawKeypoints(rgb1,kp1,imgShow,flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow("Keypoints_1",imgShow)
    cv2.waitKey(0)
    imgShow = None
    imgShow = cv2.drawKeypoints(rgb2,kp2,imgShow,flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Keypoints_2",imgShow)
    cv2.waitKey(0)
    
    FLANN_INDEX_KDTREE = 0
    

    index_params = dict(algorithm = FLANN_INDEX_KDTREE) # parameters refenence FLANN
    search_params = dict(checks = 50) #or pass empty dictionary
                                      #high values will take more time 
                                      #but gives better prescision
   
    matcher = cv2.FlannBasedMatcher(index_params,search_params)
   
    matches = matcher.match(des1,des2)
    print("Find total " + str(len(matches)) + " matches.")
    imgMatches = None
    imgMatches = cv2.drawMatches(rgb1,kp1,rgb2,kp2,matches,imgMatches)
    cv2.imshow("matches",imgMatches)
    cv2.waitKey(0)
    
    goodMatches = []
    minDis = 9999.0
    for i in range(0,len(matches)):
        if matches[i].distance < minDis:
            minDis = matches[i].distance
    for i in range(0,len(matches)):
        if matches[i].distance < (minDis * 3):
            goodMatches.append(matches[i])
    print("good matches = " + str(len(goodMatches)))
    imgMatches = None
    imgMatches = cv2.drawMatches(rgb1,kp1,rgb2,kp2,goodMatches,imgMatches)
    cv2.imshow("good_matches",imgMatches)
    cv2.waitKey(0)

    pts_obj = []
    pts_img = []
    for i in range(0,len(goodMatches)):
        p = kp1[goodMatches[i].queryIdx].pt
        d = depth1[int(p[1])][int(p[0])]
        if d == 0:
            pass
        else:
            pts_img.append(kp2[goodMatches[i].trainIdx].pt)
            
            pd = point2DTO3D(p[1],p[0],d,camera)
            
            pts_obj.append(pd)
    pts_obj = np.array(pts_obj)
    pts_img = np.array(pts_img)
    cameraMatrix = np.matrix(CameraIntrinsicData)
    rvec = None
    tvec = None
    inliers = None
    retval , rvec, tvec, inliers = cv2.solvePnPRansac(pts_obj,pts_img,cameraMatrix,distCoeffs, iterationsCount = 100, reprojectionError = 0.66)
    print("inliers: "+ str(len(inliers)))
    print("R=" + str(rvec))
    print("t=" + str(tvec))
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()

if __name__ == "__main__":
    filepath = "/home/lin/internal-save/"
    depthfile1 = filepath + 'depth3.png'
    depthfile2 = filepath + 'depth4.png'
    rgbfile1 = filepath + 'rgb3.png'
    rgbfile2 = filepath + 'rgb4.png'
    RGBFilenames = [rgbfile1,rgbfile2]
    DepthFilenames = [depthfile1,depthfile2]
    rgbs , depths  = readImgFiles(RGBFilenames,DepthFilenames,-1)
    print type(depths[0])   
    camera = CameraIntrinsicParameters(325.5,253.5,518.0,519.0,1000.0)
    CameraIntrinsicData = readyaml.parseYamlFile("./data/camera.yml")

    computeMatches(rgbs[0],rgbs[1],depths[0],depths[1],CameraIntrinsicData,None,camera)
