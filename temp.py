import cv2

url = "rtsp://172.20.10.9:554/mjpeg/1"

cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("No frame")
        continue
    
    print("Got frame")

    cv2.imshow("ESP32", frame)

    if cv2.waitKey(1) == 27:
        break