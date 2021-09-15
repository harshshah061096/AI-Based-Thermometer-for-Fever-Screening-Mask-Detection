import cv2
import numpy as np
from smbus2 import SMBus
from mlx90614 import MLX90614
with_mask=np.load('with_mask.npy')
without_mask=np.load('without_mask.npy')
with_mask=with_mask.reshape(600,50*50*3)
without_mask=without_mask.reshape(600,50*50*3)
X=np.r_[with_mask,without_mask]
labels=np.zeros(X.shape[0])
labels[600:]=1
name={0:'mask',1:'no mask'}
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,labels,test_size=0.25)
from sklearn.decomposition import PCA
pca=PCA(n_components=3)
x_train=pca.fit_transform(x_train)
svm=SVC()
svm.fit(x_train,y_train)
x_test=pca.fit_transform(x_test)
y_pred=svm.predict(x_test)
accuracy_score(y_test,y_pred)
haar_data=cv2.CascadeClassifier('data.xml')
capture=cv2.VideoCapture(0)
data=[]
bus = SMBus(1)
while True:
    flag,img=capture.read()
    sensor = MLX90614(bus, address=0x5A)
    if flag:
        faces=haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            sensor = MLX90614(bus, address=0x5A)
            face=img[y:y+h,x:x+w,:]
            face=cv2.resize(face,(50,50))
            face=face.reshape(1,-1)
            face=pca.transform(face)
            pred=svm.predict(face)[0]
            n=name[int(pred)]
            if n=='mask' and  sensor.get_object_1()<37.77:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
                cv2.rectangle(img,(x,y+w-25),(x+w,y+h),(0,255,0),cv2.FILLED)
                font =cv2.FONT_HERSHEY_DUPLEX
                result ='mask ' + str(sensor.get_object_1())
                cv2.putText(img,result,(x+6,y+w-6),font,0.5,(0,0,0),1)
            elif n== 'no mask' and sensor.get_object_1()<37.77:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),1)
                cv2.rectangle(img,(x,y+w-25),(x+w,y+h),(255,0,255),cv2.FILLED)
                font =cv2.FONT_HERSHEY_DUPLEX
                result ='no mask modrate temp' + str(sensor.get_object_1())
                cv2.putText(img,result,(x+6,y+w-6),font,0.5,(0,0,0),1)
            else :
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
                cv2.rectangle(img,(x,y+w-25),(x+w,y+h),(0,0,255),cv2.FILLED)
                font =cv2.FONT_HERSHEY_DUPLEX
                result ='mask high temp' + str(sensor.get_object_1())
                cv2.putText(img,result,(x+6,y+w-6),font,0.5,(0,0,0),1)
        cv2.imshow("x",img)
        if cv2.waitKey(2)==27:
            break
        else:
            print('0')
capture.release()
cv2.destroyAllWindows()