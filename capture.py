import cv2
import sys


faceCascade = cv2.CascadeClassifier("C:\\Users\\ACER\\Documents\hackthon-proj\haarcascade_frontalface_alt.xml")

video_capture = cv2.VideoCapture(0)
i = 0

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=0
    )

    a = 0

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:

        # put the text 
        text = "student"
        text_x = x
        text_y = y + h + 20
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255, 0), 1)

        # save the ROI
        face_roi = frame[y:y+h, x:x+w]
        cv2.imwrite(f"dataset/user.{a}.{i}.png", face_roi)
        i += 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()