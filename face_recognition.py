import cv2
import numpy as np
import os

def dist(x1,x2):
    return np.sqrt(sum(((x1-x2)**2)))

def knn(X,Y,k=5):
    val=[]
    m=X.shape[0]
    for i in range(m):
        ix=X[i,:-1]
        iy=X[i,-1]

        d=dist(Y,ix)
        val.append((d,iy))
    vals=sorted(val,key=lambda x:x[0])[:k]
    vals=np.array(vals)[:,-1]
    
    new_val=np.unique(vals,return_counts=True)
    #print(new_val)
    index=np.argmax(new_val[1])
    pred=new_val[0][index]
    return pred


cap=cv2.VideoCapture(0)

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

skip=0
face_data=[]
dataset_path='./data/'
label=[]
class_id=0
names={}

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):

        names[class_id]=fx[:-4]

        print("loaded "+fx)
        data_item=np.load(dataset_path+fx)
        face_data.append(data_item)


        #Create labels for class
        target=class_id*np.ones((data_item.shape[0],))
        class_id+=1
        label.append(target)


face_dataset=np.concatenate(face_data,axis=0)
labels_dataset=np.concatenate(label,axis=0).reshape((-1,1))

print(face_dataset.shape)
print(labels_dataset.shape)

trainset=np.concatenate((face_dataset,labels_dataset),axis=1)
print(trainset.shape)

while True:
    ret,frame=cap.read()

    if ret==False:
        continue

    faces=face_cascade.detectMultiScale(frame,1.3,5)

    for face in faces:
        x,y,w,h=face

        offset=10

        face_section=frame[y-offset:y+h+offset,x-offset:x+offset+w]
        face_section=cv2.resize(face_section,(100,100))

        out=knn(trainset,face_section.flatten())
        pred=names[int(out)]

        cv2.putText(frame,pred,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,1),2,cv2.LINE_AA)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

    cv2.imshow("frame",frame)
    key=cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()