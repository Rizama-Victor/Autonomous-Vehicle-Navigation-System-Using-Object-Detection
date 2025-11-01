import cv2
import numpy as np
import os
import time

# define the minimum confidence (to filter weak detections), Non-Maximum Suppression (NMS) threshold, the green color, and the class name
confidence_thresh = 0.5
NMS_thresh = 0.0
green = (0, 255, 0)
class_names = "data.names"

# initialize the video capture object
video_capture = cv2.VideoCapture(0)

# load the class labels the model was trained on
class_path = class_names  # unpack content of class
with open(class_path, "r") as f:
    classes = f.read().strip().split("\n")

# we will now load the configuration and weight files from disk

yolo_config = "configuration_file.cfg"
yolo_weights = "model_weights.weights"


# load the pre-trained yolo network
net = cv2.dnn.readNetFromDarknet(yolo_config, yolo_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Get the name of all the layers in the network
layer_names = net.getLayerNames()
# Get the names of the output layers
# output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
output_layers = net.getUnconnectedOutLayersNames()

# Now, we want to loop though each frame in the video stream
while True:
    # the start time to compute frame per sec (fps)
    start = time.time()
    
    # read the video frame
    success, frame = video_capture.read(0)
    frame = cv2.resize(frame, (640, 480))


    # in absence of any more frames to show, break out of the while loop
    if not success:
        break

    # Now get the frame dimensions
    # N.B: The frames are just like cut out images from the video stream since a video is a collection of moving images
    h = frame.shape[0]
    w = frame.shape[1]

    # create a blob from the image
    blob = cv2.dnn.blobFromImage(
        frame, 1 / 255, (416, 416), swapRB=True, crop=False)
    # pass the blob through the network and get the output predictions
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # create empty lists for storing the bounding boxes and confidences
    boxes = []
    confidences = []
    class_ids = []

    # loop over the output predictions
    for output in outputs:
        # loop over the detections
        for detection in output:
            # get the class ID and confidence of the detected object
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # filter out weak detections by keeping only those with a confidence
            # above the minimum confidence threshold (i.e 0.5 in this case)
            if confidence > confidence_thresh:
                # perform element-wise multiplication to get the coordinates of the bounding box
                box = [int(a * b) for a, b in zip(detection[0:4], [w, h, w, h])]
                center_x, center_y, width, height = box
                
                # get the top-left corner of the bounding box
                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))

                # append the bounding box and the confidence to their respective lists
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, width, height])
    
                # put bounding box coordinates in a list

                output_data = [class_id, x, y, width, height]
                output_data = [str(value) for value in output_data] 
                print(output_data, end=' ') # print output data in a straight line

                file_name  = "current/data2.txt" # path to text file where bounding box coordinates and class ids will be saved
                
                # code to write bounding box coordinates to the text file
                with open(file_name, "a") as file:
                    boundary_ordinates = ' '.join(output_data)
                    file.write(boundary_ordinates + "\n")

        # apply non-maximum suppression to remove weak bounding boxes that overlap with others.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thresh, NMS_thresh)
        # indices = indices.flatten()

        for i in indices:
            (x,y,w,h) = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            cv2.rectangle(frame, (x, y), (x + w, y + h), green, 2)
            text = f"{classes[class_ids[i]]}: {confidences[i] * 100:.2f}%"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 2)
            
            if len(output_data) > 1:
                # print(len(output_data))
                # save frame with bounding box
                frame_copy = frame.copy()
                cv2.imwrite(f"frames2/frame_{time.time()}.jpg", frame_copy)

        #  end time to compute the frame per seconds (i.e fps)
        end = time.time()

        # next is to calculate the fps and draw it on the frame
        fps = f"FPS: {1/(end - start):.2f}"
        cv2.putText(frame, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 8)


            # now, code to display the frame
        cv2.imshow("Frame", frame)
    # if the "q" key is pressed, the loop should instantly stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# now, release the video capture object
video_capture.release()
cv2.destroyAllWindows
                




