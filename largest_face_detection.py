import cv2
import numpy as np

cap=cv2.VideoCapture(0)
face=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
skip=0
face_data=[]
dataset_path='./data/'
file_name=input("enter the name:")
while True:
	ret,frame=cap.read()
	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	if ret==False:
		continue

	faces=face.detectMultiScale(frame,1.3,5)
	faces=sorted(faces,key=lambda f:f[2]*f[3])

	for (x,y,w,h) in faces[-1:]:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

		offset=10
		face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section=cv2.resize(face_section,(100,100))

		skip+=1
		if skip%10==0:
			face_data.append(face_section)
			print(face_data)

	

	cv2.imshow("frame",frame)
	cv2.imshow("face_section",face_section)

	key=cv2.waitKey(30) & 0xFF

	if key==ord('q'):
		break	

face_data=np.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

np.save(dataset_path+file_name+".npy",face_data)


cap.release()
cv2.destroyAllWindows()
