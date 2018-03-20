#!usr/bin/env python
# -*- codingL utf-8 -*-

import yaml,sys
import numpy as np

def fixYamlFile(filename):
    with open(filename,'rb') as f:
        lines = f.readlines()
        if lines[0] != '%YAML 1.0\n':
            lines[0] = '%YAML 1.0\n'
        for line in lines:
            if' !!opencv-matrix' in line:
              	lines[lines.index(line)] = line.split(' !!opencv-matrix')[0] + '\n'
        with open(filename,'wb') as fw:
        	fw.writelines(lines)
                    
def parseYamlFile(filename):
    fixYamlFile(filename)
    f = open(filename)
    x = yaml.load(f)
    f.close()
    arr = np.array(x['camera_matrix']['data'],dtype = np.float32)
    distortionCoeff = np.array(x['distortion_coefficients']['data'],dtype = np.float32)

    return arr.reshape(3,3) , distortionCoeff 
if __name__ == '__main__':
    print(parseYamlFile(sys.argv[1]))

