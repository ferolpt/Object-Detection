import cv2


cam = cv2.VideoCapture("highway.mp4")

#Object detection from Stable camara
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, frame = cam.read()
    height, width, _ = frame.shape

    #print(height, width)
    #720 1280

    #Extract Region of interest
    roi = frame [300:720, 650:1000]

    #object detection
    mask = object_detector.apply(roi)

    # mask 0-255 0:black 255:complete white
    _, mask = cv2.threshold(mask, 254,255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        #calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 200:
            #cv2.drawContours(frame, [cnt], -1, (0,255,0),2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x,y), (x+w,y+h),(0,255,0),3)
    
    cv2.imshow("Roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(30) == ord("q"):
        break

cam.release()

cv2.destroyAllWindows()