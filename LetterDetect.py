'''

@author: Anthony_2
'''
import cv2
import numpy as np
import sys
import operator
'''Following taken from example code found online'''
# module level variables ##########################################################################
MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

###################################################################################################
class ContourWithData():

    # member variables ############################################################################
    npaContour = None           # contour
    boundingRect = None         # bounding rect for contour
    intRectX = 0                # bounding rect top left corner x location
    intRectY = 0                # bounding rect top left corner y location
    intRectWidth = 0            # bounding rect width
    intRectHeight = 0           # bounding rect height
    fltArea = 0.0               # area of contour

    def calculateRectTopLeftPointAndWidthAndHeight(self):               # calculate bounding rect info
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):                            # this is oversimplified, for a production grade program
        if self.fltArea < MIN_CONTOUR_AREA: return False        # much better validity checking would be necessary
        return True

###################################################################################################
'''
'''
# Get the ROI cropped with the letter of interest, assumes that rotation has been applied
def getROI(bthresh):
    bThreshCopy = bthresh.copy()

    imgContours, npaContours, npaHierarchy = cv2.findContours(bThreshCopy,              # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                 cv2.RETR_EXTERNAL,                 # retrieve the outermost contours only
                                                 cv2.CHAIN_APPROX_NONE)           # compress horizontal, vertical, and diagonal segments and leave only their end points

    cv2.imshow("Contours",imgContours)
    cv2.imshow("BThresh",bthresh)
    cv2.waitKey(15)

    print len(npaContours)

    largestContour = npaContours[0]
    contourArea = cv2.contourArea(largestContour)
    print(contourArea)

    for npaContour in npaContours:                          # for each contour
        if cv2.contourArea(npaContour) > contourArea:
            largestContour = npaContour
            contourArea = cv2.contourArea(npaContour)
            print(contourArea)
    
        #cv2.waitKey(50)

    [intX, intY, intW, intH] = cv2.boundingRect(largestContour)         # get and break out bounding rect
    letterROI = bthresh[intY:intY+intH, intX:intX+intW]                                  # crop char out of threshold image
    cv2.imshow("imgROI", letterROI)                    # show cropped out char for reference

    cv2.waitKey(500)
    return letterROI

def getLetter(letterROI):
    allContoursWithData = []                # declare empty lists,
    validContoursWithData = []              # we will fill these shortly
    
    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)                  # read in training classifications
    except:
        print "error, unable to open classifications.txt, exiting program\n"
        sys.exit
    # end try
    
    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)                 # read in training images
    except:
        print "error, unable to open flattened_images.txt, exiting program\n"
        sys.exit
    # end try
    
    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))       # reshape numpy array to 1d, necessary to pass to call to train
    
    kNearest = cv2.ml.KNearest_create()                   # instantiate KNN object
    
    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
    
    imgTestingNumbers = letterROI
    '''
    imgGray = cv2.cvtColor(imgTestingNumbers, cv2.COLOR_BGR2GRAY)       # get grayscale image
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)                    # blur
    
                                                        # filter image from grayscale to black and white
    imgThresh = cv2.adaptiveThreshold(imgBlurred,                           # input image
                                      255,                                  # make pixels that pass the threshold full white
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # use gaussian rather than mean, seems to give better results
                                      cv2.THRESH_BINARY_INV,                # invert so foreground will be white, background will be black
                                      11,                                   # size of a pixel neighborhood used to calculate threshold value
                                      2)                                    # constant subtracted from the mean or weighted mean
    '''
    
    imgThreshCopy = letterROI.copy()        # make a copy of the thresh image, this in necessary b/c findContours modifies the image
    
    imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,             # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                 cv2.RETR_EXTERNAL,         # retrieve the outermost contours only
                                                 cv2.CHAIN_APPROX_SIMPLE)   # compress horizontal, vertical, and diagonal segments and leave only their end points
    
    for npaContour in npaContours:                             # for each contour
        contourWithData = ContourWithData()                                             # instantiate a contour with data object
        contourWithData.npaContour = npaContour                                         # assign contour to contour with data
        contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)     # get the bounding rect
        contourWithData.calculateRectTopLeftPointAndWidthAndHeight()                    # get bounding rect info
        contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)           # calculate the contour area
        allContoursWithData.append(contourWithData)                                     # add contour with data object to list of all contours with data
    # end for
    
    for contourWithData in allContoursWithData:                 # for all contours
        if contourWithData.checkIfContourIsValid():             # check if valid
            validContoursWithData.append(contourWithData)       # if so, append to valid contour list
        # end if
    # end for
    
    validContoursWithData.sort(key = operator.attrgetter("intRectX"))         # sort contours from left to right
    
    
    for contourWithData in validContoursWithData:            # for each contour
                                                # draw a green rect around the current char
        cv2.rectangle(imgTestingNumbers,                                        # draw rectangle on original testing image
                     (contourWithData.intRectX, contourWithData.intRectY),     # upper left corner
                     (contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),      # lower right corner
                     (0, 255, 0),              # green
                     2)                        # thickness
    
        imgROI = letterROI[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,     # crop char out of threshold image
                           contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]
    
        imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))             # resize image, this will be more consistent for recognition and storage
    
        npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))      # flatten image into 1d numpy array
    
        npaROIResized = np.float32(npaROIResized)       # convert from 1d numpy array of ints to 1d numpy array of floats
    
        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)     # call KNN function find_nearest
    
        strCurrentChar = str(chr(int(npaResults[0][0])))                                             # get character from results
        print(strCurrentChar)
        return strCurrentChar
    # end for

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

t=cv2.imread('ScaleTarget.png')
b,g,r = cv2.split(t)

retr, rthresh = cv2.threshold(r, 100, 255, cv2.THRESH_BINARY) 
retb, bthresh = cv2.threshold(b, 100, 255, cv2.THRESH_BINARY) 

keyblobs = detector.detect(r)

tBlobbed = cv2.drawKeypoints(t, keyblobs, np.array([]), (255,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
rBlobbed = cv2.drawKeypoints(r, keyblobs, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
rthBlobbed = cv2.drawKeypoints(rthresh, keyblobs, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

letterROI = getROI(bthresh)

'''
'''
getLetter(letterROI)
