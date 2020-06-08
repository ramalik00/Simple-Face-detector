import cv2
import numpy as np
import argparse

ap=argparse.ArgumentParser()
ap.add_argument("-i","--input",type=str,default="")
ap.add_argument("-d","--display",type=int,default="1")
ap.add_argument("-o","--output",type=str,default="")
args=vars(ap.parse_args())


img=cv2.imread(args["input"])
resized=cv2.resize(img,(300,300))
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray_img, 1.05, 4)
for x,y,w,h in faces:
    img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)

resized=cv2.resize(img,(int(img.shape[0]),int(img.shape[1])))
if args['display']>0:
   cv2.imshow("image",resized)
   cv2.waitKey(0)
   cv2.destroyAllWindows()

if args["output"]!="":
    
    cv2.imwrite(args["output"],resized)
   

                   

