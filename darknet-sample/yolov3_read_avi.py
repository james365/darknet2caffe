import time
from timeit import default_timer as timer
import numpy as np
import cv2
import os


def video_demo():

    weightsPath = "./yolov3_tiny.weights"
    configPath = "./voc.cfg"
    labelsPath = "./voc.names"

    LABELS = open(labelsPath).read().strip().split("\n") 
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8") 
    boxes = []    
    confidences = []    
    classIDs = []    
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    
    video_path="test25.avi"
    capture = cv2.VideoCapture(video_path)
    length = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    
    prev_time =0
    accum_time = 0
    curr_fps = 0
    fps=""
    i=0
    while (i<length):
        i+=1
        prev_time=timer()
        ref, src = capture.read()
        if(ref==False):
           #time.sleep(0.01)
           continue
           
        sp = src.shape
        
        width=sp[1]
        height=sp[0]
        chan=sp[2]

        image=src.copy()
        
        (H, W) = image.shape[:2]
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]        
        
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)
        
        for output in layerOutputs:
            
            for detection in output:
                scores = detection[5:]                
                classID = np.argmax(scores)
                confidence = scores[classID]                
                
                if confidence > 0.5:
                    
                    
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    
                    
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    
                    
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
                    
                    if(len(boxes)>16):
                      del boxes[0]
                      del confidences[0]
                      del classIDs[0]
                    
                    
                    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.3)
                    i=max(idxs.flatten())
                    
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    
                    
                    color = [int(c) for c in COLORS[classIDs[i]]] 
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)

                    text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                    #print(text)
                    


        
        
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time += exec_time
        
        curr_fps += 1
        if accum_time > 1:
            
            fps=curr_fps
            
            accum_time -= 1
            curr_fps = 0
        fps_str = "FPS:" + str(fps)
        print(fps_str)
        cv2.putText(image, fps_str, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.imshow("Image", image)
        
        c = cv2.waitKey(30) & 0xff
        if c == 27:
           capture.release()
           break
video_demo()