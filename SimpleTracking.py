# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 12:27:58 2018
@author: Kunal Sarkar
#Automatic Vehicle Movement Detection
"""
import cv2
import numpy as np
path='Full Real Uk Practical Driving Test Pass Video.mp4'
FrameSkip=7
ResetPoint1=20
ResetPoint2=30
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1065,480))
cap = cv2.VideoCapture(path)
font = cv2.FONT_HERSHEY_SIMPLEX
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))
t=0
# Take first frame and find corners in it and remove initial unnecessary frames
while(t<260):
    t=t+1
    ret, old_frame = cap.read() 
#croped for the roi
old_frame=old_frame[20:500,35:1100,:]   
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
#initial feature point
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
#cap = cv2.VideoCapture(path)
t=0
while(t<=36729):#
    t=t+1
   # print(t)
    ret,frame1 = cap.read()
    frame=frame1[20:500,35:1100,:]
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if(t%ResetPoint1==0):
        mask = np.zeros_like(old_frame)
#    cv2.imshow('out1',frame_gray)
    if(t%ResetPoint2==0):
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)  
    
#    # calculate optical flow
    if(t%FrameSkip==0):
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        dst = np.square(np.subtract(good_new,good_old))
        dst1=np.sqrt(np.sum(dst,axis=1))
        dst1=np.sort(dst1)
        print(t)
        print(np.sum(dst1[int(dst1.shape[0]/2):]))
        if(np.sum(dst1[:int(dst1.shape[0]/2)])>=dst1.shape[0]):
            cv2.putText(frame1,'Vehicle is Moving',(400,100), font, 1,(0,255,0),4,cv2.LINE_AA)
        else:
            cv2.putText(frame1,'Vehicle is Stationary',(400,100), font, 1,(0,0,255),4,cv2.LINE_AA)
#    # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(frame,mask)
 #       cv2.imshow('OP',frame1)
        cv2.imshow('frame',img)
        out.write(img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
    out.write(frame_gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()
out.release()

cv2.waitKey(0)
cv2.destroyAllWindows()

