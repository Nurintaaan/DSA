import numpy as np
import cv2


class Car:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height


    def update(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height


    def setTime(self, time):
        self.time = time


    def getTime(self):
        return self.time


    def addTime(self):
        self.time += 1


    def retrieve(self):
        return self.x, self.y, self.width, self.height


# cap = cv2.VideoCapture('inputs/1300/UI_Pengambilan2_01.avi')
cap = cv2.VideoCapture('inputs/1700/UI_Conv_P2_06.avi')
# cap = cv2.VideoCapture('inputs/2100/00025_1.mp4')

fbg = cv2.bgsegm.createBackgroundSubtractorMOG()
# fbg = cv2.createBackgroundSubtractorMOG2()

# get frame size
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# create a mask (manual for each camera)
mask = np.zeros((frame_h,frame_w), np.uint8)
mask[:,:] = 255
mask[:100, :] = 0
mask[230:, 160:190] = 0
mask[170:230,170:190] = 0
mask[140:170,176:190] = 0
mask[100:140,176:182] = 0
CONTOUR_WIDTH = 30
CONTOUR_HEIGHT = 30
LINE_THICKNESS = 1
TRAFFIC_TIME_THRESHOLD = 50
TRAFFIC_CAR_THRESHOLD = 5

DISTANCE = 5

objects = {}
counter = 0
currentTime = 0
currentCar = 0

while (True):
    ret, frame = cap.read()
    fmask = fbg.apply(frame, 0.1)
    kernel = np.ones((5, 5), np.uint8)
    # background = fbg.getBackgroundImage(frame)

    img_dilatation = cv2.dilate(fmask, kernel, iterations=2)
    img_erosion = cv2.erode(img_dilatation, kernel, iterations=3)

    img_final = img_erosion

    threshold = cv2.bitwise_and(img_erosion, img_erosion, mask=mask)

    _, contours, hierarchy = cv2.findContours(threshold,
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        for (i, contour) in enumerate(contours):
            (x, y, width, height) = cv2.boundingRect(contour)
            contour_valid = (width > CONTOUR_WIDTH) and (height > CONTOUR_HEIGHT)

            if not contour_valid:
                continue

            isRegistered = False
            locatedAt = "None"
            for distance in range(DISTANCE):
                temp = y - distance
                # print("Checking at " + str(temp) + str(objects))
                if (objects.get(str(temp)) != None):
                    locatedAt = str(temp)
                    isRegistered = True

            if (isRegistered):
                # print(locatedAt + str(objects.get(locatedAt)))
                car = objects.pop(locatedAt)
                car.update(x, y, width, height)
                car.addTime()
                currentTime = car.getTime()
                # print("Traffic indicator: " + str(currentTime))
                currentCar = len(objects)
                if (currentTime > TRAFFIC_TIME_THRESHOLD and currentCar > TRAFFIC_CAR_THRESHOLD):
                    print("Traffic Jam Alert!")
                # print("Updating car at " + locatedAt)
                objects[str(y)] = car
            else:
                counter += 1
                car = Car(x, y, width, height)
                car.setTime(1)
                objects[str(y)] = car
                # print("New car added! Total= " + str(counter))
                # print("x={:d}, y={:d}, w={:d}, h={:d}".format(x, y, width, height))

            cv2.rectangle(frame, (x, y), (x + width - 1, y + height - 1), (255, 255, 255), LINE_THICKNESS)

    cv2.imshow('frame', frame)
    # cv2.imshow('subtraction', img_final)
    # cv2.imshow('frame', background)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
