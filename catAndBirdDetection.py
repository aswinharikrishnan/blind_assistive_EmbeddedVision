import cv2
import numpy as np
import urllib.request
import pyttsx3
import time

# ESP32 camera URL
url = "http://192.168.10.76/cam-hi.jpg"

# speech engine
engine = pyttsx3.init()

whT = 416
confThreshold = 0.5
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


def detectObjects(img):

    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    detected_objects = []

    blob = cv2.dnn.blobFromImage(img, 1/255, (whT, whT), [0,0,0], 1, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(outputNames)

    for output in outputs:
        for det in output:

            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > confThreshold:

                w = int(det[2] * wT)
                h = int(det[3] * hT)

                x = int((det[0] * wT) - w / 2)
                y = int((det[1] * hT) - h / 2)

                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    if len(indices) > 0:

        for i in indices:

            i = int(i)

            x,y,w,h = bbox[i]

            label = classNames[classIds[i]]
            conf = confs[i]

            detected_objects.append(label)

            print(f"Detected: {label} | Confidence: {conf:.2f}")

            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)

            cv2.putText(
                img,
                f"{label.upper()} {int(conf*100)}%",
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,0,255),
                2,
            )

    return detected_objects


while True:

    try:
        img_resp = urllib.request.urlopen(url)
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        img = cv2.imdecode(img_np, -1)

    except Exception as e:
        print("Camera connection error:", e)
        continue

    objects = detectObjects(img)

    # SPEAK DETECTED OBJECTS
    if len(objects) > 0:

        sentence = " , ".join(objects)

        speech = f"I see {sentence}"

        print("Speaking:", speech)

        engine.say(speech)
        engine.runAndWait()

    else:

        print("No object detected")
        engine.say("No object detected")
        engine.runAndWait()

    cv2.imshow("ESP32 CAM Detection", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    # wait 5 seconds before next detection
    time.sleep(5)

cv2.destroyAllWindows()

# speak in every detection
# import cv2
# import numpy as np
# import urllib.request
# import pyttsx3
# import time

# # ESP32 camera URL
# url = "http://192.168.10.76/cam-hi.jpg"

# engine = pyttsx3.init()

# whT = 320
# confThreshold = 0.5
# nmsThreshold = 0.3

# # load class names
# classesFile = "coco.names"
# classNames = []

# with open(classesFile, "rt") as f:
#     classNames = f.read().rstrip("\n").split("\n")

# # load YOLO
# modelConfig = "yolov3.cfg"
# modelWeights = "yolov3.weights"

# net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# def findObjects(outputs, img):

#     hT, wT, cT = img.shape
#     bbox = []
#     classIds = []
#     confs = []

#     for output in outputs:
#         for det in output:

#             scores = det[5:]
#             classId = np.argmax(scores)
#             confidence = scores[classId]

#             if confidence > confThreshold:

#                 w = int(det[2] * wT)
#                 h = int(det[3] * hT)

#                 x = int((det[0] * wT) - w / 2)
#                 y = int((det[1] * hT) - h / 2)

#                 bbox.append([x, y, w, h])
#                 classIds.append(classId)
#                 confs.append(float(confidence))

#     indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

#     if len(indices) > 0:

#         for i in indices:

#             i = int(i)

#             x, y, w, h = bbox[i]

#             label = classNames[classIds[i]]
#             conf = confs[i]

#             # print detection
#             print(f"Detected: {label} | Confidence: {conf:.2f}")

#             # speak every detection
#             speech_text = f"{label} detected"

#             engine.say(speech_text)
#             engine.runAndWait()

#             # draw bounding box
#             cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

#             cv2.putText(
#                 img,
#                 f"{label.upper()} {int(conf*100)}%",
#                 (x, y - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.6,
#                 (255, 0, 255),
#                 2,
#             )

#             # small pause to avoid speech overlap
#             time.sleep(0.2)


# while True:

#     try:
#         img_resp = urllib.request.urlopen(url)
#         img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
#         img = cv2.imdecode(img_np, -1)
#     except:
#         print("Camera connection lost")
#         continue

#     blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
#     net.setInput(blob)

#     layerNames = net.getLayerNames()
#     outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]

#     outputs = net.forward(outputNames)

#     findObjects(outputs, img)

#     cv2.imshow("ESP32-CAM Detection", img)

#     if cv2.waitKey(1) & 0xFF == 27:
#         break

cv2.destroyAllWindows()
# camera not connecting issue
# import cv2
# import numpy as np
# import urllib.request
# import pyttsx3

# # ESP32 camera URL
# url = "http://10.15.52.237/cam-hi.jpg"

# # initialize speech engine
# engine = pyttsx3.init()

# # last spoken object
# last_spoken = ""

# whT = 416
# confThreshold = 0.3
# nmsThreshold = 0.3

# # load class names
# classesFile = "coco.names"
# classNames = []

# with open(classesFile, "rt") as f:
#     classNames = f.read().rstrip("\n").split("\n")

# # load YOLO
# modelConfig = "yolov3.cfg"
# modelWeights = "yolov3.weights"

# net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# def findObjects(outputs, img):

#     global last_spoken

#     hT, wT, cT = img.shape
#     bbox = []
#     classIds = []
#     confs = []

#     for output in outputs:
#         for det in output:

#             scores = det[5:]
#             classId = np.argmax(scores)
#             confidence = scores[classId]

#             if confidence > confThreshold:

#                 w = int(det[2] * wT)
#                 h = int(det[3] * hT)

#                 x = int((det[0] * wT) - w / 2)
#                 y = int((det[1] * hT) - h / 2)

#                 bbox.append([x, y, w, h])
#                 classIds.append(classId)
#                 confs.append(float(confidence))

#     indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

#     if len(indices) > 0:

#         for i in indices:

#             i = int(i)

#             x, y, w, h = bbox[i]

#             label = classNames[classIds[i]]
#             conf = confs[i]

#             print(f"Detected: {label} | Confidence: {conf:.2f}")

#             # speak if object changed
#             if label != last_spoken:

#                 text = f"{label} detected"
#                 engine.say(text)
#                 engine.runAndWait()

#                 last_spoken = label

#             # draw box
#             cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

#             cv2.putText(
#                 img,
#                 f"{label.upper()} {int(conf*100)}%",
#                 (x, y - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.6,
#                 (255, 0, 255),
#                 2,
#             )


# while True:

#     try:
#         img_resp = urllib.request.urlopen(url)
#         img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
#         img = cv2.imdecode(img_np, -1)
#     except:
#         print("Camera connection lost")
#         continue

#     blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
#     net.setInput(blob)

#     layerNames = net.getLayerNames()
#     outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]

#     outputs = net.forward(outputNames)

#     findObjects(outputs, img)

#     cv2.imshow("ESP32 CAM Detection", img)

#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cv2.destroyAllWindows()

# first obj correctly detected 
# import cv2
# import numpy as np
# import urllib.request
# import pyttsx3

# # ESP32 camera URL
# url = "http://192.168.10.76/cam-hi.jpg"

# engine = pyttsx3.init()

# whT = 320
# confThreshold = 0.5
# nmsThreshold = 0.3

# # store objects from previous frame
# previous_objects = set()

# # load class names
# classesFile = "coco.names"
# classNames = []

# with open(classesFile, "rt") as f:
#     classNames = f.read().rstrip("\n").split("\n")

# # load YOLO
# modelConfig = "yolov3.cfg"
# modelWeights = "yolov3.weights"

# net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# def findObjects(outputs, img):

#     global previous_objects

#     hT, wT, cT = img.shape
#     bbox = []
#     classIds = []
#     confs = []

#     current_objects = set()

#     for output in outputs:
#         for det in output:

#             scores = det[5:]
#             classId = np.argmax(scores)
#             confidence = scores[classId]

#             if confidence > confThreshold:

#                 w = int(det[2] * wT)
#                 h = int(det[3] * hT)

#                 x = int((det[0] * wT) - w / 2)
#                 y = int((det[1] * hT) - h / 2)

#                 bbox.append([x, y, w, h])
#                 classIds.append(classId)
#                 confs.append(float(confidence))

#     indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

#     if len(indices) > 0:

#         for i in indices:

#             i = int(i)

#             x, y, w, h = bbox[i]

#             label = classNames[classIds[i]]
#             conf = confs[i]

#             current_objects.add(label)

#             print(f"Detected: {label} | Confidence: {conf:.2f}")

#             # speak only if object is NEW
#             if label not in previous_objects:

#                 text = f"{label} detected"
#                 engine.say(text)
#                 engine.runAndWait()

#             # draw box
#             cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

#             cv2.putText(
#                 img,
#                 f"{label.upper()} {int(conf*100)}%",
#                 (x, y - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.6,
#                 (255, 0, 255),
#                 2,
#             )

#     # update previous objects
#     previous_objects = current_objects


# while True:

#     try:
#         img_resp = urllib.request.urlopen(url)
#         img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
#         img = cv2.imdecode(img_np, -1)
#     except:
#         print("Camera connection lost")
#         continue

#     blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
#     net.setInput(blob)

#     layerNames = net.getLayerNames()
#     outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]

#     outputs = net.forward(outputNames)

#     findObjects(outputs, img)

#     cv2.imshow("ESP32-CAM Detection", img)

#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cv2.destroyAllWindows()

# telling person detected once only
# import cv2
# import numpy as np
# import urllib.request
# import pyttsx3

# # ESP32 camera URL
# url = "http://192.168.10.76/cam-lo.jpg"

# # ESP32 buzzer trigger
# esp32_alert = "http://192.168.10.76/yolo"

# # speech engine
# engine = pyttsx3.init()

# # store spoken objects
# spoken_objects = set()

# whT = 320
# confThreshold = 0.5
# nmsThreshold = 0.3

# # load class names
# classesFile = "coco.names"
# classNames = []

# with open(classesFile, "rt") as f:
#     classNames = f.read().rstrip("\n").split("\n")

# # load YOLO
# modelConfig = "yolov3.cfg"
# modelWeights = "yolov3.weights"

# net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# def findObjects(outputs, img):

#     global spoken_objects

#     hT, wT, cT = img.shape
#     bbox = []
#     classIds = []
#     confs = []

#     current_objects = set()

#     for output in outputs:
#         for det in output:

#             scores = det[5:]
#             classId = np.argmax(scores)
#             confidence = scores[classId]

#             if confidence > confThreshold:

#                 w = int(det[2] * wT)
#                 h = int(det[3] * hT)

#                 x = int((det[0] * wT) - w / 2)
#                 y = int((det[1] * hT) - h / 2)

#                 bbox.append([x, y, w, h])
#                 classIds.append(classId)
#                 confs.append(float(confidence))

#     indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

#     if len(indices) > 0:

#         for i in indices:

#             i = int(i)

#             x, y, w, h = bbox[i]

#             label = classNames[classIds[i]]
#             conf = confs[i]

#             current_objects.add(label)

#             print(f"Detected: {label} | Confidence: {conf:.2f}")

#             # speak only if object not spoken before
#             if label not in spoken_objects:

#                 text = f"{label} detected"

#                 engine.say(text)
#                 engine.runAndWait()

#                 spoken_objects.add(label)

#             # draw box
#             cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

#             cv2.putText(
#                 img,
#                 f"{label.upper()} {int(conf*100)}%",
#                 (x, y - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.6,
#                 (255, 0, 255),
#                 2,
#             )

#     # reset spoken objects if scene changes
#     if len(current_objects) == 0:
#         spoken_objects.clear()


# while True:

#     try:
#         img_resp = urllib.request.urlopen(url)
#         img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
#         img = cv2.imdecode(img_np, -1)
#     except:
#         print("Camera connection lost")
#         continue

#     blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
#     net.setInput(blob)

#     layerNames = net.getLayerNames()
#     outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]

#     outputs = net.forward(outputNames)

#     findObjects(outputs, img)

#     cv2.imshow("ESP32-CAM YOLO Detection", img)

#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cv2.destroyAllWindows()


#detect print and read aloud

# import cv2
# import numpy as np
# import urllib.request
# import requests
# import pyttsx3

# # ESP32 camera URL
# url = "http://192.168.10.76/cam-lo.jpg"

# # ESP32 buzzer trigger
# esp32_alert = "http://192.168.10.76/yolo"

# # Voice engine
# engine = pyttsx3.init()

# last_object = ""

# whT = 320
# confThreshold = 0.5
# nmsThreshold = 0.3

# # Load class names
# classesFile = "coco.names"
# classNames = []

# with open(classesFile, "rt") as f:
#     classNames = f.read().rstrip("\n").split("\n")

# # Load YOLO
# modelConfig = "yolov3.cfg"
# modelWeights = "yolov3.weights"

# net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# def findObjects(outputs, img):

#     global last_object

#     hT, wT, cT = img.shape
#     bbox = []
#     classIds = []
#     confs = []

#     for output in outputs:
#         for det in output:

#             scores = det[5:]
#             classId = np.argmax(scores)
#             confidence = scores[classId]

#             if confidence > confThreshold:

#                 w = int(det[2] * wT)
#                 h = int(det[3] * hT)

#                 x = int((det[0] * wT) - w / 2)
#                 y = int((det[1] * wT) - h / 2)

#                 bbox.append([x, y, w, h])
#                 classIds.append(classId)
#                 confs.append(float(confidence))

#     indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

#     if len(indices) > 0:

#         for i in indices:

#             i = int(i)

#             x, y, w, h = bbox[i]

#             label = classNames[classIds[i]]
#             conf = confs[i]

#             print(f"Detected: {label} | Confidence: {conf:.2f}")

#             # Speak only when object changes
#             if label != last_object:

#                 text = f"{label} detected"

#                 engine.say(text)
#                 engine.runAndWait()

#                 last_object = label

#                 # trigger ESP32 buzzer
#                 try:
#                     requests.get(esp32_alert)
#                 except:
#                     pass

#             # Draw bounding box
#             cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

#             cv2.putText(
#                 img,
#                 f"{label.upper()} {int(conf*100)}%",
#                 (x, y - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.6,
#                 (255, 0, 255),
#                 2,
#             )


# while True:

#     try:
#         img_resp = urllib.request.urlopen(url)
#         img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
#         img = cv2.imdecode(img_np, -1)
#     except:
#         print("Camera connection lost")
#         continue

#     blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
#     net.setInput(blob)

#     layerNames = net.getLayerNames()
#     outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]

#     outputs = net.forward(outputNames)

#     findObjects(outputs, img)

#     cv2.imshow("ESP32-CAM YOLO Detection", img)

#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cv2.destroyAllWindows()


# #detect object and print on terminal
# import cv2
# import numpy as np
# import urllib.request
# import requests

# # ESP32 camera stream
# url = "http://192.168.10.76/cam-hi.jpg"

# # ESP32 buzzer API
# esp32_alert = "http://192.168.10.76/yolo"

# whT = 320
# confThreshold = 0.5
# nmsThreshold = 0.3

# # Load class names
# classesFile = "coco.names"
# classNames = []

# with open(classesFile, "rt") as f:
#     classNames = f.read().rstrip("\n").split("\n")

# # Load YOLO model
# modelConfig = "yolov3.cfg"
# modelWeights = "yolov3.weights"

# net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# def findObjects(outputs, img):

#     hT, wT, cT = img.shape
#     bbox = []
#     classIds = []
#     confs = []

#     for output in outputs:
#         for det in output:

#             scores = det[5:]
#             classId = np.argmax(scores)
#             confidence = scores[classId]

#             if confidence > confThreshold:

#                 w = int(det[2] * wT)
#                 h = int(det[3] * hT)

#                 x = int((det[0] * wT) - w / 2)
#                 y = int((det[1] * hT) - h / 2)

#                 bbox.append([x, y, w, h])
#                 classIds.append(classId)
#                 confs.append(float(confidence))

#     indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

#     if len(indices) > 0:

#         for i in indices:

#             i = int(i)

#             x, y, w, h = bbox[i]

#             label = classNames[classIds[i]]
#             conf = confs[i]

#             # PRINT OBJECT + CONFIDENCE
#             print(f"Detected: {label} | Confidence: {conf:.2f}")

#             # Send alert to ESP32
#             try:
#                 requests.get(esp32_alert)
#             except:
#                 pass

#             # Draw box
#             cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

#             cv2.putText(
#                 img,
#                 f"{label.upper()} {int(conf*100)}%",
#                 (x, y - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.6,
#                 (255, 0, 255),
#                 2,
#             )


# while True:

#     try:
#         img_resp = urllib.request.urlopen(url)
#         img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
#         img = cv2.imdecode(img_np, -1)
#     except:
#         print("Camera connection lost")
#         continue

#     blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
#     net.setInput(blob)

#     layerNames = net.getLayerNames()
#     outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]

#     outputs = net.forward(outputNames)

#     findObjects(outputs, img)

#     cv2.imshow("ESP32-CAM Detection", img)

#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cv2.destroyAllWindows()



#detect any object

# import cv2
# import numpy as np
# import urllib.request

# # ESP32 CAM URL
# url = 'http://192.168.10.76/cam-hi.jpg'

# whT = 320
# confThreshold = 0.5
# nmsThreshold = 0.3

# # Load class names
# classesfile = 'coco.names'
# classNames = []

# with open(classesfile, 'rt') as f:
#     classNames = f.read().rstrip('\n').split('\n')

# # Load YOLO
# modelConfig = 'yolov3.cfg'
# modelWeights = 'yolov3.weights'

# net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# def findObject(outputs, im):

#     hT, wT, cT = im.shape
#     bbox = []
#     classIds = []
#     confs = []

#     found_cat = False
#     found_bird = False

#     for output in outputs:
#         for det in output:

#             scores = det[5:]
#             classId = np.argmax(scores)
#             confidence = scores[classId]

#             if confidence > confThreshold:

#                 w = int(det[2] * wT)
#                 h = int(det[3] * hT)

#                 x = int((det[0] * wT) - w / 2)
#                 y = int((det[1] * hT) - h / 2)

#                 bbox.append([x, y, w, h])
#                 classIds.append(classId)
#                 confs.append(float(confidence))

#     indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

#     if len(indices) > 0:

#         for i in indices:

#             i = int(i)

#             box = bbox[i]
#             x, y, w, h = box

#             label = classNames[classIds[i]]

#             if label == 'bird':
#                 found_bird = True

#             if label == 'cat':
#                 found_cat = True

#             cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 255), 2)

#             cv2.putText(im,
#                         f'{label.upper()} {int(confs[i]*100)}%',
#                         (x, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.6,
#                         (255, 0, 255),
#                         2)

#     if found_cat and found_bird:
#         print("ALERT: Cat and Bird detected")


# while True:

#     try:
#         # Read image from ESP32 CAM
#         img_resp = urllib.request.urlopen(url)
#         imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
#         im = cv2.imdecode(imgnp, -1)

#     except:
#         print("Camera connection lost")
#         continue

#     blob = cv2.dnn.blobFromImage(im, 1/255, (whT, whT), [0,0,0], 1, crop=False)

#     net.setInput(blob)

#     layernames = net.getLayerNames()
#     outputNames = [layernames[i - 1] for i in net.getUnconnectedOutLayers()]

#     outputs = net.forward(outputNames)

#     findObject(outputs, im)

#     cv2.imshow("ESP32-CAM Detection", im)

#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cv2.destroyAllWindows()

#only 2 objects-bird and cat

# import cv2
# import numpy as np
# import urllib.request

# # ESP32-CAM image URL
# url = 'http://192.168.10.76/cam-hi.jpg'

# whT = 320
# confThreshold = 0.5
# nmsThreshold = 0.3

# classesfile = 'coco.names'
# classNames = []

# with open(classesfile, 'rt') as f:
#     classNames = f.read().rstrip('\n').split('\n')

# modelConfig = 'yolov3.cfg'
# modelWeights = 'yolov3.weights'

# net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# def findObject(outputs, im):

#     hT, wT, cT = im.shape

#     bbox = []
#     classIds = []
#     confs = []

#     found_cat = False
#     found_bird = False

#     for output in outputs:
#         for det in output:

#             scores = det[5:]
#             classId = np.argmax(scores)
#             confidence = scores[classId]

#             if confidence > confThreshold:

#                 w = int(det[2] * wT)
#                 h = int(det[3] * hT)

#                 x = int((det[0] * wT) - w / 2)
#                 y = int((det[1] * hT) - h / 2)

#                 bbox.append([x, y, w, h])
#                 classIds.append(classId)
#                 confs.append(float(confidence))

#     indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

#     if len(indices) > 0:

#         for i in indices.flatten():

#             box = bbox[i]

#             x, y, w, h = box

#             if classNames[classIds[i]] == 'bird':
#                 found_bird = True

#             elif classNames[classIds[i]] == 'cat':
#                 found_cat = True

#             if classNames[classIds[i]] == 'bird':

#                 cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 255), 2)

#                 cv2.putText(
#                     im,
#                     f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
#                     (x, y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.6,
#                     (255, 0, 255),
#                     2
#                 )

#                 print('bird')
#                 print(found_bird)

#             if classNames[classIds[i]] == 'cat':

#                 cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 255), 2)

#                 cv2.putText(
#                     im,
#                     f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
#                     (x, y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.6,
#                     (255, 0, 255),
#                     2
#                 )

#                 print('cat')
#                 print(found_cat)

#     if found_cat and found_bird:
#         print('alert')


# while True:

#     try:

#         img_resp = urllib.request.urlopen(url)

#         imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)

#         im = cv2.imdecode(imgnp, -1)

#         blob = cv2.dnn.blobFromImage(
#             im, 1/255, (whT, whT), [0, 0, 0], 1, crop=False
#         )

#         net.setInput(blob)

#         layernames = net.getLayerNames()

#         outputNames = [layernames[i - 1] for i in net.getUnconnectedOutLayers()]

#         outputs = net.forward(outputNames)

#         findObject(outputs, im)

#         cv2.imshow('Image', im)

#     except Exception as e:
#         print("Camera Error:", e)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()