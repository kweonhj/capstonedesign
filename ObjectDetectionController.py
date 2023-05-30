#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 15:12:44 2023

@author: kweonhyuckjin
"""

import cv2
import torch
import time
import numpy as np
from PID import PID
from yolov5 import YOLOv5
from MagneticFieldCalculator import MagneticFieldCalculator
from CalculateActuationMatrix import CalculateActuationMatrix
from ArduinoSetting import ArduinoSetting

class ObjectDetectionController:
    
    def __init__(self, custom_weights_path, camera_indices, width, height, P, I, D, setpoint_xy, setpoint_z, ard, desired_B):
        self.custom_weights_path = custom_weights_path
        self.camera_indices = camera_indices
        self.width = width
        self.height = height
        self.P = P
        self.I = I
        self.D = D
        self.setpoint_xy = setpoint_xy
        self.setpoint_z = setpoint_z
        self.model = None
        self.camera_streams = None
        self.prev_time = None
        self.fps_interval = 10
        self.frame_count = 0
        self.feedback_value_xy = None
        self.feedback_value_z = None
        self.output_xy = None
        self.output_z = None
        self.location = None
        self.output = None
        self.pid_xy = None
        self.pid_z = None
        self.ard = ard
        self.desired_B = desired_B
        
    def run(self):
        # Load the YOLOv5 model with custom weights
        self.model = YOLOv5(self.custom_weights_path)

        # Define camera streams
        self.camera_streams = [cv2.VideoCapture(index) for index in self.camera_indices]

        self.prev_time = time.time()

        # Create PID controllers
        self.pid_xy = PID(self.P, self.I, self.D, self.setpoint_xy)
        self.pid_z = PID(self.P, self.I, self.D, self.setpoint_z)
        
        ArduinoControl = ArduinoSetting(self.ard)
        ArduinoControl.arduino_setting()
        
        
        while True:
            frames = []
            desired_I = np.random.rand(6,1)
            start_x1 = 1920 * 4.3/26.3
            start_x2 = 1920 * 2.3/26.3
            limit_x1 = 1920 * 17.7 / 26.3
            limit_x2 = 1920 * 17.5/ 26.3 
            limit_y1 = 1080 * 12 / 15 
            limit_y2 = 1080 * 12.2 / 15 
            
            for camera_stream_index, camera_stream in enumerate(self.camera_streams):

                # camera_stream.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                # camera_stream.set(28, 70)
                
                ret, frame = camera_stream.read()
                # Resize frame width for only detecting in workspace and put the name on frame 
                if camera_stream_index == 0:
                    frame = frame[int(0):int(limit_y1), int(start_x1):int(limit_x1)]  
                    cv2.putText(frame, 'TOP VIEW',(800, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                elif camera_stream_index == 1:
                    frame = frame[int(0):int(limit_y2), int(start_x2):int(limit_x2)]    
                    cv2.putText(frame, 'SIDE VIEW',(900, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                    
                if not ret:
                    break
                

                #frame = cv2.resize(frame, (self.width, self.height))
                frame_height, frame_width, _ = frame.shape
                x_range = [0, frame_width]
                y_range = [0, frame_height]

                results = self.model.predict(frame)

                self.frame_count += 1
                if self.frame_count % self.fps_interval == 0:
                    curr_time = time.time()
                    elapsed_time = curr_time - self.prev_time
                    fps = self.fps_interval / elapsed_time
                    self.prev_time = curr_time
                    print(f"FPS: {fps:.2f}")

                for detection in results.xyxy[0]:
                    obj_class = int(detection[5])
                    class_name = 'ROBOT'
                    confidence = detection[4]
                    bbox = detection[:4].tolist()
                    
                    if confidence > 0.5:
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'{class_name}: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                        centroid_x = (x1 + x2) / 2
                        centroid_y = (y1 + y2) / 2

                        if camera_stream_index == 0:
                            self.feedback_value_xy = [centroid_x, centroid_y]
                            normalized_x = (self.feedback_value_xy[0] - x_range[0]) / (x_range[1] - x_range[0])
                            normalized_y = (self.feedback_value_xy[1] - y_range[0]) / (y_range[1] - y_range[0])
                            self.feedback_value_xy = [normalized_x, normalized_y]
                            self.output_xy = self.pid_xy.update(self.feedback_value_xy)
                            
                        elif camera_stream_index == 1:
                            self.feedback_value_z = centroid_y
                            normalized_z = (self.feedback_value_z - y_range[0]) / (y_range[1] - y_range[0])
                            self.feedback_value_z = [normalized_z]
                            self.output_z = self.pid_z.update(self.feedback_value_z)
                            
                if self.feedback_value_xy is not None and self.feedback_value_z is not None:
                    
                    self.location = np.concatenate((self.feedback_value_xy, self.feedback_value_z))
                    self.output = np.concatenate((self.output_xy, self.output_z))
                    
                    position = (self.location + np.array([-0.5, -0.5, -0.5])) * np.array([0.0065/0.5, 0.0083/0.5, 0.007/0.5]) 
                    print('position :')
                    print(position * 1e3)
                    # currents???
                    app = CalculateActuationMatrix(self.desired_B, position)
                    Actuation_matrix = app.calculate()

                    desired_F = self.output
                    desired_B = self.desired_B
                    BF = np.concatenate((desired_B, desired_F))
                    BF = BF.flatten()
                    desired_I = np.dot(np.linalg.pinv(Actuation_matrix), BF)
                    random_vector = np.random.uniform(low=-0.2, high=0.2, size= 8)
                    # I0 = [0 ,0 ,0 ,0, 0, 0, 0, 0]
                    ArduinoControl.MotordriveControl(random_vector)
                    # self.feedback_value_xy = None
                    # self.feedback_value_z = None
                    
                else:
                    
                    I0 = [0 ,0 ,0 ,0, 0, 0, 0, 0]
                    ArduinoControl.MotordriveControl(I0)
                    
                cv2.imshow(f'Camera {self.camera_indices[camera_stream_index]}', frame)
                
                

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        for camera_stream in self.camera_streams:
            camera_stream.release()

        cv2.destroyAllWindows()



