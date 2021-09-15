
import cv2
import matplotlib.pyplot as plt
haar_data=cv2.CascadeClassifier('data.xml')
capture=cv2.VideoCapture(0)
data=[]
print ("without mask training")
while True:
    flag,img=capture.read()
    if flag:
        faces=haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
            face=img[y:y+h,x:x+w,:]
            face=cv2.resize(face,(50,50))
            print(len(data))
            if len(data)<600:
                data.append(face)
        cv2.imshow("x",img)
        if cv2.waitKey(2)==27 or len(data) >= 600:
            break
        else:
            print('0')
capture.release()
cv2.destroyAllWindows()
import numpy as np
np.save('without_mask.npy',data)


capture=cv2.VideoCapture(0)
data=[]
print ("with mask training")
while True:
    flag,img=capture.read()
    if flag:
        faces=haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
            face=img[y:y+h,x:x+w,:]
            face=cv2.resize(face,(50,50))
            print(len(data))
            if len(data)<600:
                data.append(face)
        cv2.imshow("x",img)
        if cv2.waitKey(2)==27 or len(data) >= 600:
            break
        else:
            print('0')
capture.release()
cv2.destroyAllWindows()
np.save('with_mask.npy',data)