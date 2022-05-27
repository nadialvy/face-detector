import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0) #capture video from webcam
                    # 0 is default webcam
                    # kalau mau letakin video juga bisa

#iterate over frames in the video
while True:
    # read the current frame
    successful_frame_read, frame = webcam.read()

    # convert the frame to grayscale
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscale_img)
    
    # draw rectangle around the face
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(128, 256), randrange(128, 256), randrange(128, 256)), 8)

    cv2.imshow('Face Detector', frame)
    cv2.waitKey(1) #mengizinkan user untuk menampilkan window selama bbrp mili (ditentukan) atau selama keyboard apapun ditekan
                    #kalau diberikan 0 sbg parameter maka menunggu user untuk menekan tombol apapun habis itu baru nutup -> dalam kasus image
                    # kalau diberikan 1 sbg parameter maka gausah tekan tombol apapun, maka nunggu 1milisecond akan secara otomatis perfi ke next iteration



