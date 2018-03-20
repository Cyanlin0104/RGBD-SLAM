from __future__ import print_function
import os


path = '/home/lin/RGBD_slam/data/'
dist_path = path + 'calib/'

if not os.path.exists(dist_path):
	os.makedirs(dist_path,mode=0755)
	print ('created directory : %s'% dist_path)

files = os.listdir(path)

s=[]

count = 0
for file in files:

	if 'cal' in file and not os.path.isdir(file):
		print (file)
		os.rename(path + file, dist_path + file)
		count += 1

print ('{} files moved'.format(count), end="\n")
