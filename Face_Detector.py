import cv2

# Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Choose an image to detect faces in
img = cv2.imread('aespa.jpg')

# convert the image into grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect faces                          multi scale artinya bisa mendeteksi wajah ntah besar atau kecil
face_coordinates = trained_face_data.detectMultiScale(gray_img)
 
# draw rectangle around the face  
for (x, y, w, h) in face_coordinates:      
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

# print(face_coordinates)

# Show image
cv2.imshow('Face Detector', img)

# wait until the key is pressed, presss any key to exit
cv2.waitKey()

print("Code completed")