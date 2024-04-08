import re
import numpy as np
import cv2
from skimage import restoration
from skimage.feature import blob_doh



def nothing(x):
    pass

def rgb(*data):
    return np.array(data)

cv2.namedWindow('image')


cv2.createTrackbar('param1','image',0,100,nothing)
cv2.createTrackbar('param2','image',0,100,nothing)
cv2.createTrackbar('param3','image',0,100,nothing)

cv2.setTrackbarPos('param1', 'image', 5)
cv2.setTrackbarPos('param2', 'image', 2)
cv2.setTrackbarPos('param3', 'image', 2)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')



#number signifies camera
cap = cv2.VideoCapture(2)



def detect_eyes(img):
    eyes = eye_cascade.detectMultiScale(img, 1.3, 5) # detect eyes
    width = np.size(img, 1) # get face frame width
    height = np.size(img, 0) # get face frame height
    left_eye = None
    right_eye = None

    for (x, y, w, h) in eyes:
        if y > height / 2:
            pass
        eyecenter = x + w / 2  # get the eye center
        if eyecenter < width * 0.5:
            left_eye = ((x, y, w, h), img[y:y + h, x:x + w])
        else:
            right_eye = ((x, y, w, h), img[y:y + h, x:x + w])
    return left_eye, right_eye

def detect_faces(img):
    coords = face_cascade.detectMultiScale(img, 1.3, 5)
    frame = None
    i = None
    if len(coords) > 1:
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None
    for (x, y, w, h) in biggest:
        frame = ((x, y, w, h), img[y:y + h, x:x + w])
        
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
    return frame

def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width]  # cut eyebrows out (15 px)
    return img



def getPupilCenterData(img):
    
    param1 = cv2.getTrackbarPos('param1','image')
    param2 = cv2.getTrackbarPos('param2','image')
    param3 = cv2.getTrackbarPos('param3','image')

    
    faceData = detect_faces(img)
    
    if faceData is None:
        return None
    
    (fx, fy, fw, fh), face = faceData
    
    eyes = detect_eyes(face)
    
    result = []
    for eyeData in eyes:
        
        if eyeData is None:
            result.append(None)
            continue
        
        (ex, ey, ew, eh), eye = eyeData
        ex += fx
        ey += fy
        _, eye = cv2.threshold(eye, 0, 255, cv2.THRESH_BINARY)

        Y, X = np.where(eye==255)
        Z = np.column_stack((X,Y)).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, param1, param2)
        nClusters = 1
        ret,label,center=cv2.kmeans(Z,nClusters,None,criteria,param3,cv2.KMEANS_RANDOM_CENTERS)
        
        
        cX, cY = center[0]
        cX += ex
        cY += ey
        
        result.append((int(cX), int(cY)))
        
    return result        

left_means = []
right_means = []

while 1:
    ret, img = cap.read()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    if len(left_means) > 100:
        del left_means[-1]
    if len(right_means) > 100:
        del right_means[-1]
    
    centers = getPupilCenterData(gray)
    
    if centers is None:
        continue
    
    left, right = centers
    
    if left is not None:
        left_means.append(left)
    
    xMean, yMean = np.array(left_means).mean(axis=0)
    cv2.circle(img, (xMean, yMean), 5, (0,255,0), -1)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
    cv2.imshow('image',img)


cap.release()
cv2.destroyAllWindows()
