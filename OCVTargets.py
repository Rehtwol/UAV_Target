'''
Modified on Feb 15, 2016

Parts borrowed from:
https://codeplasma.com/2012/12/03/getting-webcam-images-with-python-and-opencv-2-for-real-this-time/

@author: Anthony_2
'''

import cv2
import numpy as np
from cv2 import bitwise_not, bitwise_and
 
# Camera 0 is the default camera
camera_port = 0
#Number of frames to throw away while the camera adjusts to light levels
ramp_frames = 30
# Initialize the camera capture object
camera = cv2.VideoCapture(camera_port)

# Captures a single image from the camera
def get_image():
    retval, im = camera.read()
    return im

# Ramp the camera - these frames will be discarded
for i in xrange(ramp_frames):
    temp = get_image()
    cv2.imshow("Ramp",temp)
    cv2.waitKey(25)

# Start of targeting code
print("Starting targeting")


# Parameters for blob detection
detParam = cv2.SimpleBlobDetector_Params()

detParam.filterByArea = True
detParam.filterByCircularity = True
detParam.filterByColor = True
detParam.filterByConvexity = False
detParam.filterByInertia = True
detParam.blobColor = 255
detParam.minArea = 6000
detParam.maxArea = 600000
detParam.minThreshold = 90
detParam.maxThreshold = 150
detParam.minCircularity = 0.1
detParam.minInertiaRatio = 0.5


detector = cv2.SimpleBlobDetector_create(detParam)

# Flag and counter for exiting loop
onTarget = False
counter = 0
while not onTarget:
    # Get image and split it up
    t = get_image()
    #t=cv2.imread('ScaleTarget.png')
    b,g,r = cv2.split(t)
    
    #cv2.imwrite("TestB"+str(counter)+".png", b)
    #cv2.imwrite("TestG"+str(counter)+".png", g)
    #cv2.imwrite("TestR"+str(counter)+".png", r)
    
    #rnot = bitwise_not(r)
    
    # Thresheld images for each band
    retr, rthresh = cv2.threshold(r, 100, 255, cv2.THRESH_BINARY) 
    #retg, gthresh = cv2.threshold(g, 127, 255, cv2.THRESH_BINARY) 
    retb, bthresh = cv2.threshold(b, 100, 255, cv2.THRESH_BINARY) 
    
    # Initial detection and drawing detected blobs into images
    keyblobs = detector.detect(r)
    #keyblobs = detector.detect(b)
    
    
    if keyblobs:
        print(keyblobs)
        keyptx = keyblobs[0].pt[0]
        keypty = keyblobs[0].pt[1]
        keyptsz = keyblobs[0].size
        print("X=" + str(keyptx) +" Y=" + str(keypty) + " Size=" + str(keyptsz))
        #tBlobbed = cv2.drawKeypoints(t, keyblobs, np.array([]), (255,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #rBlobbed = cv2.drawKeypoints(r, keyblobs, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #rthBlobbed = cv2.drawKeypoints(rthresh, keyblobs, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
     
    
    tBlobbed = cv2.drawKeypoints(t, keyblobs, np.array([]), (255,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    rBlobbed = cv2.drawKeypoints(r, keyblobs, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    rthBlobbed = cv2.drawKeypoints(rthresh, keyblobs, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #bBlobbed = cv2.drawKeypoints(b, keyblobs, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #bthBlobbed = cv2.drawKeypoints(bthresh, keyblobs, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
     
    rth2, rcontours, rhierarchy = cv2.findContours(rthresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    print(rcontours)
    rthCont = cv2.drawContours(rthresh, rcontours, -1, (255,255,255), 3)
     
    #cv2.imwrite("Blobbed"+str(counter)+".png", tBlobbed)
    cv2.imshow("Blobs",tBlobbed)
    #cv2.imshow("Red",rthresh)
    cv2.waitKey(20)
    cv2.imshow("RBlobs",rBlobbed)
    #cv2.imshow("Green",gthresh)
    #cv2.waitKey(20)
    cv2.imshow("RThresh",rthBlobbed)
    cv2.imshow("RContour",rthresh)
    #cv2.imshow("BlueChan",b)
    #cv2.imshow("BThresh",bthresh)
    #cv2.imshow("R&G",rgthresh)
    #cv2.waitKey(15)
    
    
    if counter == 11:
        cv2.imwrite("Blobbed"+str(counter)+".png", tBlobbed)
        cv2.imwrite("BlobbedRed"+str(counter)+".png", rBlobbed)
        cv2.imwrite("BlobbedRedThresh"+str(counter)+".png", rthBlobbed)
        cv2.imwrite("ContourRedThresh"+str(counter)+".png",rthCont)
        cv2.imwrite("TestB"+str(counter)+".png", b)
        cv2.imwrite("TestG"+str(counter)+".png", g)
        cv2.imwrite("TestR"+str(counter)+".png", r)
        cv2.imwrite("t"+str(counter)+".png", t)
        #cv2.imwrite("BlobbedBlue"+str(counter)+".png", bBlobbed)
        #cv2.imwrite("BlobbedBlueThresh"+str(counter)+".png", bthBlobbed)


#        print(counter)
    
    counter += 1
'''    if counter>3000:
        onTarget = True
'''