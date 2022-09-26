import streamlink
import cv2

carsData = cv2.CascadeClassifier("cars.xml")
pedestriansData = cv2.CascadeClassifier("pedestrians.xml")

streams = streamlink.streams("https://www.earthcam.com/usa/newyork/timessquare/index.php?cam=tsstreet")
stream_url = streams["best"].url

webcam = cv2.VideoCapture(stream_url)

cv2.namedWindow("Face Detection")

while True:
    
    sucessful_frame_read, frame = webcam.read()

    grayScale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cars_coordinates = carsData.detectMultiScale(grayScale)
    pedestrians_cordinates = pedestriansData.detectMultiScale(grayScale)

    for (x, y, w, h) in cars_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

    for (x, y, w, h) in pedestrians_cordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)

    cv2.imshow("Face Detection", frame)
    key = cv2.waitKey(1)

    if(key == 27):
        break