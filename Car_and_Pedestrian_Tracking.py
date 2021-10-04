import cv2
from random import randrange

#our image
img_file = 'Car Image.jpg'
#video=cv2.VideoCapture('Tesla Dashcam Accident.mp4')
video=cv2.VideoCapture('Pedestrians Compilation.mp4')

#our pretrained data
car_classifier= 'cars.xml'
pedestrian_classifier='haarcascade_fullbody.xml'

#create car classifier
car_tracker = cv2.CascadeClassifier(car_classifier)
pedestrian_tracker= cv2.CascadeClassifier(pedestrian_classifier)

while True:
    #create opencv image
    (read_successful, frame) = video.read()
    #Safe Code
    if read_successful:
        #convert to grayscale
        grayscaled_frame= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break


    #detect cars and pedestrians
    cars= car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians= pedestrian_tracker.detectMultiScale(grayscaled_frame)


    #draw rectangles around cars and pedestrians
    for(x,y,w,h) in cars:
       cv2.rectangle(frame,(x,y),(x+w, y+h), (0,0,randrange(0,256)), 2)
      # cv2.rectangle(frame,(x,y),(x+w, y+h), (randrange(0,256),0,0), 2)
    for(x,y,w,h) in pedestrians:
       cv2.rectangle(frame,(x,y),(x+w, y+h), (31,95,255), 2)  


    #Display the image
    cv2.imshow('CP Car detector',frame)
    key= cv2.waitKey(1)

    #quit button
    if key==81 or key==113:
       break



video.release()


'''

#print coordinates of car
#print(cars)



#Display the image
cv2.imshow('CP Car detector',img)
cv2.waitKey()

'''
print("Code completed")