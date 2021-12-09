import cv2

Vid = cv2.VideoCapture(0)
face_cas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
jas=cv2.imread("Jason_mask.png")
while True:
    ret, frame = Vid.read()
    face=face_cas.detectMultiScale(frame,scaleFactor=1.2,minNeighbors=4)
    for (x, y, w, h) in face:
        jas_siz=cv2.resize(jas,(w,h))
        _,face_mask=cv2.threshold(cv2.cvtColor(jas_siz,cv2.COLOR_BGR2GRAY),25,255,cv2.THRESH_BINARY_INV)
        face_rect=frame[y:y+h,x:x+w]
        op=cv2.bitwise_and(face_rect,face_rect,mask=face_mask)
        plus=cv2.add(op,jas_siz)
        frame[y:y + h, x:x + w]=plus
    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
