from glob import glob 
from random import sample 
import numpy as np
import cv2
from random import sample 
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
np.set_printoptions(suppress=True)
import plotly.graph_objects as go

from triangulate import * 


random.seed(2022) #CV1
random.seed(2018) #CV2
random.seed(2011) #cv3
random.seed(2007) #CV4
random.seed(2022) #CV5


#### CGI Triangulation ####

files = random.sample(glob(r"C:\Users\rocs\Documents\avci_mt\eval\*.txt"),20)
c = 1

for file in files:
    if c >5 :
        break
    i = file[37:-4]
    print("\n",c,". sample \n",i)
    
    #get predictions
    left_p,right_p = get_pred(file)
    if left_p.shape[0] == 10 & right_p.shape[0]==10 : # show only 10 kwires
        
        tri_pred = triangulate_cgi(left_p.reshape(20,1,2),right_p.reshape(20,1,2),  Baseline_pos = 1)
        print('\nPredicted triangulation\n')
        print(repr(tri_pred))
    
        # show predictions
        im = cv2.imread(file[:-4]+str('.png'))
        cv2.imwrite(r"..\\results_cgi\\"+ i + ".png", im)
        cv2.imshow("pred",im)
        cv2.waitKey(0)
        
        #3d viz
        #xp,yp,zp = tri_pred[:,0], tri_pred[:,1], tri_pred[:,2]
        #fig = go.Figure(data=[go.Scatter3d(x=xp, y=yp, z=zp, mode='markers')])
        #fig.write_html("htmls/pred"+str(i)+".html")
        
        c = c+1
        
    else:
        print("skipped")
    
    
    
#### REALK DATA Triangulation####

random.seed(2022) #CV1
random.seed(2018) #CV2
random.seed(2011) #cv3
random.seed(2007) #CV4
random.seed(2022) #CV5

files = random.sample(glob(r"..\eval\cv_5\*.txt"),20)
c = 1

for file in files:
    if c >5:
        break
    i = file[-27:-4]
    print("\n",c,". sample \n",i)
    
    #get predictions
    left_p,right_p = get_pred(file)
    if left_p.shape[0] == 6 & right_p.shape[0]==6 :
        
        tri_pred = triangulate(left_p.reshape(12,1,2),right_p.reshape(12,1,2),  Baseline_pos = 0)
        print('\nPredicted triangulation\n')
        print(repr(tri_pred))
                
        # show predictions
        im = cv2.imread(file[:-4]+str('.png'))
        #cv2.imwrite(r"..\\results\\"+ i + ".png", im)
        cv2.imshow("pred",im)
        cv2.waitKey(0)
        
        # 3d viz 
        #xp,yp,zp = tri_pred[:,0], tri_pred[:,1], tri_pred[:,2]
        #fig = go.Figure(data=[go.Scatter3d(x=xp, y=yp, z=zp, mode='markers')])
        #fig.write_html("htmls/pred"+str(i)+".html")

        c = c+1
        
    else:
        print("skipped")
    