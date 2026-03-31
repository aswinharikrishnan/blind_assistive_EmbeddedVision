#correct code speaks every 5 sec 
import cv2
import numpy as np
import urllib.request
import pyttsx3
import time

# ESP32 CAM URL
url = "http://192.168.10.76/cam-lo.jpg"

whT = 416
confThreshold = 0.4
nmsThreshold = 0.3

# load class names
classesFile = "coco.names"
classNames = []

with open(classesFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# load YOLO
modelConfig = "yolov3.cfg"
modelWeights = "yolov3.weights"

net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def speak(text):
    engine = pyttsx3.init('sapi5')
    engine.setProperty('rate',150)
    engine.setProperty('volume',1.0)

    engine.say(text)
    engine.runAndWait()


def detect_objects(img):

    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    detected = []

    blob = cv2.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],1,crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(outputNames)

    for output in outputs:
        for det in output:

            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > confThreshold:

                w = int(det[2]*wT)
                h = int(det[3]*hT)

                x = int((det[0]*wT)-w/2)
                y = int((det[1]*hT)-h/2)

                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)

    if len(indices) > 0:

        for i in indices:

            i = int(i)

            x,y,w,h = bbox[i]

            label = classNames[classIds[i]]
            conf = confs[i]

            detected.append(label)

            print(f"Detected: {label} | Confidence: {conf:.2f}")

            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)

            cv2.putText(img,
                        f"{label.upper()} {int(conf*100)}%",
                        (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255,0,255),
                        2)

    return detected


while True:

    try:
        img_resp = urllib.request.urlopen(url)
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        img = cv2.imdecode(img_np, -1)

    except:
        print("Camera connection lost")
        continue

    objects = detect_objects(img)

    if len(objects) > 0:

        unique_objects = list(dict.fromkeys(objects))
        sentence = ", ".join(unique_objects)

        speech = f"I see {sentence}"

        print("Speaking:", speech)

        speak(speech)

    else:

        print("No object detected")
        speak("No object detected")

    cv2.imshow("ESP32 CAM Detection", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    time.sleep(5)

cv2.destroyAllWindows()