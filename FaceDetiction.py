import cv2

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')

cap=cv2.VideoCapture(0)
id=input("enter your id: ")
count=0
while True:

    ret,img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        count+=1
        cv2.imwrite("dataset/User." + str(id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=img[y:y+h,x:x+w]
        eyes=eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes :
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('camera',img)
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= 30:
         break

cap.release()
cv2.destroyAllWindows()


