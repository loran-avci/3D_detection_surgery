
from glob import glob 
from random import sample 
import numpy as np
import cv2
from random import sample 
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import pyplot as plt


def get_pred(file):
    pos = np.loadtxt(file,delimiter =",")
    lpos = pos[:,:4].copy()
    rpos = pos.copy()
    rpos[:,0:1] = rpos[:,0:1] - rpos[:,4:5] 
    rpos[:,2:3] = rpos[:,2:3] - rpos[:,4:5] 
    rpos = rpos[:,:4]
    rpos = rpos
    left = lpos
    right = rpos
    return left, right


def get_gt(file_l, w = 1280, h = 720):
    """get ground truth data"""
    #paths
    left =  file_l[:-5] + "L.txt"
    right = file_l[:-5] + "R.txt"
    # left
    pos = np.loadtxt(left,delimiter =",")
    x1 = pos[:,0]
    y1 = pos[:,1]
    x2 = pos[:,2]
    y2 = pos[:,3]
    p2dl = np.array([x1,y1,x2,y2])
    # right
    pos = np.loadtxt(right,delimiter =",")
    x1 = pos[:,0]
    y1 = pos[:,1]
    x2 = pos[:,2]
    y2 = pos[:,3]
    p2dr = np.array([x1,y1,x2,y2])

    x1wirel = p2dl[0,]*w
    y1wirel = p2dl[1,]*h
    x2wirel = p2dl[2,]*w
    y2wirel = p2dl[3,]*h
    x1wirer = p2dr[0,]*w
    y1wirer = p2dr[1,]*h
    x2wirer = p2dr[2,]*w
    y2wirer = p2dr[3,]*h
    
    return np.array([x1wirel,y1wirel,x2wirel,y2wirel]).T.reshape(12,1,2), np.array([x1wirer,y1wirer,x2wirer,y2wirer]).T.reshape(12,1,2)
    
    


def get_gt_cgi(file_l, w = 640, h = 320):
    """get ground truth data"""
    #paths
    left =  file_l[:-5] + "L.txt"
    right = file_l[:-5] + "R.txt"
    # left
    pos = np.loadtxt(left,delimiter =",")
    x1 = pos[:,0]
    y1 = pos[:,1]
    x2 = pos[:,2]
    y2 = pos[:,3]
    p2dl = np.array([x1,y1,x2,y2])
    # right
    pos = np.loadtxt(right,delimiter =",")
    x1 = pos[:,0]
    y1 = pos[:,1]
    x2 = pos[:,2]
    y2 = pos[:,3]
    p2dr = np.array([x1,y1,x2,y2])

    x1wirel = p2dl[0,]*w
    y1wirel = p2dl[1,]*h
    x2wirel = p2dl[2,]*w
    y2wirel = p2dl[3,]*h
    x1wirer = p2dr[0,]*w
    y1wirer = p2dr[1,]*h
    x2wirer = p2dr[2,]*w
    y2wirer = p2dr[3,]*h
    
    return np.array([x1wirel,y1wirel,x2wirel,y2wirel]).T, np.array([x1wirer,y1wirer,x2wirer,y2wirer]).T
    

def triangulate(left,right,  Baseline_pos = 0):

    #[LEFT_CAM_HD]
    fxl = 697.37
    fyl = 697.37
    cxl = 639.5
    cyl = 372.144
    k1l = -0.170949
    k2l = 0.0259925
    p1l = -0.000912223
    p2l = 0.00031079
    k3l = 0.000147782
    
    #[RIGHT_CAM_HD]
    fxr = 696.652
    fyr = 696.652
    cxr = 625.46
    cyr = 349.069
    k1r = -0.173269
    k2r = 0.0293752
    p1r = -0.000912223
    p2r = 0.00031079
    k3r = -0.00119529
    
    # Baseline
    B = 62.9438
    
    distl = np.array([k1l,k2l,p1l,p2l,k3l])
    distr = np.array([k1r,k2r,p1r,p2r,k3r])
    
    # Camera Intrinsics
    kl = np.array([[fxl,  0.0, cxl ],
                   [0.0, fyl,  cyl ],
                   [0.0, 0.0, 1.0]])
    kr = np.array([[fxr,  0.0, cxr ],
                   [0.0, fyr,  cyr ],
                   [0.0, 0.0, 1.0]])
    
    # Extrinsic Parameters
    Rtl = np.array([[1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0]])
    
    Rtr = np.array([[1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0]])
    
    
    # Projection matrix
    # Position where Baseline should be added x = 0, y = 1, z = 2
    
    Rtr[Baseline_pos,3] = -B
    projMatrl = np.dot(kl,Rtl)
    projMatrr = np.dot(kr,Rtr)
   
    
    points1u = cv2.undistortPoints(left, kl, distl, None, kl)
    points2u = cv2.undistortPoints(right, kr, distr, None, kr)
    
    points4d = cv2.triangulatePoints(projMatrl, projMatrr, points1u, points2u)
    
    points3d_undist = (points4d[:3, :]/points4d[3, :]).T
    return points3d_undist



def triangulate_cgi(left,right,  Baseline_pos = 0):

    # Baseline
    B = 63.0
    cx = 320
    cy = 240
    
    # Camera Intrinsics
    kl = np.array([[50.0,  0.0, cx ],
                   [0.0, 50.0,  cy ],
                   [0.0, 0.0, 1.0]])
    kr = np.array([[50.0,  0.0, cx ],
                   [0.0, 50.0,  cy ],
                   [0.0, 0.0, 1.0]])
    
    # Extrinsic Parameters
    Rtl = np.array([[1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0]])
    
    Rtr = np.array([[1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0]])
        
    Rtr[Baseline_pos,3] = -B
    projMatrl = np.dot(kl,Rtl)
    projMatrr = np.dot(kr,Rtr)
    #projMatrr[Baseline_pos,3] = B

    points4d = cv2.triangulatePoints(projMatrl, projMatrr, left, right)
    
    points3d_undist = (points4d[:3, :]/points4d[3, :]).T
    return points3d_undist

