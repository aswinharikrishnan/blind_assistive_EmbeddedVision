import cv2
import numpy as np
import urllib.request
import pyttsx3
import time
import threading
import argparse

class CameraStreamer:
    def __init__(self, url):
        self.url = url

    def get_frame(self):
        try:
            # Added timeout to prevent the script from hanging indefinitely if the network drops
            img_resp = urllib.request.urlopen(self.url, timeout=3)
            img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            return cv2.imdecode(img_np, -1)
        except Exception as e:
            print(f"Camera connection error: {e}")
            return None

class AlertManager:
    def speak(self, text):
        # Spawns a background thread so pyttsx3 does not freeze the video stream
        def run_tts():
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        
        thread = threading.Thread(target=run_tts)
        thread.daemon = True
        thread.start()

class ObjectDetector:
    def __init__(self, config, weights, names_file, whT=416, conf_threshold=0.5, nms_threshold=0.3):
        self.whT = whT
        self.confThreshold = conf_threshold
        self.nmsThreshold = nms_threshold
        
        with open(names_file, "rt") as f:
            self.classNames = f.read().rstrip("\n").split("\n")
            
        self.net = cv2.dnn.readNetFromDarknet(config, weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect_and_draw(self, img):
        hT, wT, cT = img.shape
        bbox, classIds, confs, detected_objects = [], [], [], []

        blob = cv2.dnn.blobFromImage(img, 1/255, (self.whT, self.whT), [0,0,0], 1, crop=False)
        self.net.setInput(blob)
        
        layerNames = self.net.getLayerNames()
        outputNames = [layerNames[i - 1] for i in self.net.getUnconnectedOutLayers()]
        outputs = self.net.forward(outputNames)

        for output in outputs:
            for det in output:
                scores = det[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                
                if confidence > self.confThreshold:
                    w, h = int(det[2] * wT), int(det[3] * hT)
                    x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                    bbox.append([x, y, w, h])
                    classIds.append(classId)
                    confs.append(float(confidence))

        indices = cv2.dnn.NMSBoxes(bbox, confs, self.confThreshold, self.nmsThreshold)

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = bbox[i]
                label = self.classNames[classIds[i]]
                conf = confs[i]
                detected_objects.append(label)

                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
                cv2.putText(img, f"{label.upper()} {int(conf*100)}%", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        return detected_objects

def main():
    # Eliminates hardcoded configurations. Run via terminal: python script.py --ip 192.168.10.76
    parser = argparse.ArgumentParser(description="ESP32-CAM YOLO Object Detection")
    parser.add_argument('--ip', type=str, required=True, help="IP address of ESP32")
    args = parser.parse_args()

    stream_url = f"http://{args.ip}/cam-hi.jpg"

    streamer = CameraStreamer(stream_url)
    detector = ObjectDetector("yolov3.cfg", "yolov3.weights", "coco.names")
    alerter = AlertManager()

    last_spoken_time = 0

    while True:
        img = streamer.get_frame()
        if img is None:
            time.sleep(1) # Back off on network failure
            continue

        objects = detector.detect_and_draw(img)

        current_time = time.time()
        # Throttles the audio to only trigger once every 3 seconds to avoid thread stacking
        if len(objects) > 0 and (current_time - last_spoken_time) > 3:
            unique_objects = list(set(objects)) # Removes duplicate words if two of the same object are found
            sentence = " and ".join(unique_objects)
            alerter.speak(f"I see {sentence}")
            last_spoken_time = current_time

        cv2.imshow("ESP32-CAM Detection", img)

        # Press ESC to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()