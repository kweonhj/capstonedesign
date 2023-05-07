#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 15:35:34 2023

@author: kweonhyuckjin
"""

import cv2
import torch
import time

# Path to your custom weights file
custom_weights_path = '/Users/kweonhyuckjin/opt/anaconda3/envs/capstonedesign/yolov5/yolov5s.pt'

# Load the YOLOv5 model with custom weights
model = torch.hub.load('ultralytics/yolov5', 'custom', path=custom_weights_path)

# Set device (GPU or CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device).eval()

# Define camera streams
camera_indices = [0, 1]  # Replace with the desired camera indices or video file paths
camera_streams = [cv2.VideoCapture(index) for index in camera_indices]

prev_time = time.time()
fps_interval = 10  # Calculate FPS every 10 frames
frame_count = 0

# Loop over camera streams
while True:
    frames = []
    
    for camera_stream in camera_streams:
        # Read frame from the camera stream
        ret, frame = camera_stream.read()
        if not ret:
            break
        
        # Perform object detection on the frame
        results = model(frame)
        frame_count += 1
        if frame_count % fps_interval == 0:
            curr_time = time.time()
            elapsed_time = curr_time - prev_time
            fps = fps_interval / elapsed_time
            prev_time = curr_time
            print(f"FPS: {fps:.2f}")
            
        # Process the detection results as per your requirements
        for detection in results.xyxy[0]:
            # Get object class, confidence, and bounding box coordinates
            obj_class = int(detection[5])
            confidence = detection[4]
            bbox = detection[:4].tolist()

            if confidence > 0.5:
                # Draw bounding box on the frame
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{obj_class}: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the frame with bounding boxes
        cv2.imshow(f'Camera {camera_indices[camera_streams.index(camera_stream)]}', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera streams and close windows
for camera_stream in camera_streams:
    camera_stream.release()

cv2.destroyAllWindows()
