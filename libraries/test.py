import os
import cv2
import time
import shutil
import numpy as np

from ultralytics import YOLO
from filterpy.kalman import KalmanFilter

import libraries.common as cv

def main(model_name, video_name, device='cpu'):
    
    model = YOLO(cv.MODELS_DIR + model_name)
    path_to_video = cv.VIDEOS_DIR + video_name
    
    video = cv2.VideoCapture(path_to_video)
    
    # Creating a Kalman Filter (used to predict the drone's trajectory)
    myKalmanFilter = KalmanFilter(dim_x=4, dim_z=2)
    myKalmanFilter.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
    myKalmanFilter.H = np.array([[1, 0, -10, 1], [0, 1, 1, -10]])
    myKalmanFilter.R *= 100. 
    myKalmanFilter.P[2:, 2:] *= 1000.
    
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    drone_positions = []
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(cv.TESTS_DIR + 'output.mp4', fourcc, fps, (frame_width, frame_height))
    
    start_time = time.time()
    
    frame_counter = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_counter += 1
        print(f"frame: {frame_counter}")
        
        # detect the drone and get its position
        drone_position = detect_drone(frame, model, device=device)
        if drone_position is not None:
            # predict the drone's position
            myKalmanFilter.predict()
            # update the drone's position
            myKalmanFilter.update(np.array(drone_position[:2]))
            # draw the bounding box, the predicted position and the trajectory
            draw_bounding_box_and_trajectory(frame, myKalmanFilter.x, drone_position, drone_positions)
    
        out.write(frame)
        
    end_time = time.time()
    print(f"total time: {end_time - start_time}")
    
    video.release()
    out.release()


def mid_point(value1, value2):
    return int((value1 + value2) / 2)

def draw_bounding_box_and_trajectory(frame, predicted_position, actual_position, drone_positions):

    # draw bounding box
    cv2.rectangle(frame, (int(actual_position[0]), int(actual_position[1])), 
                (int(actual_position[2]), int(actual_position[3])), (0, 0, 255), 2)
    
    # draw 'drone' label
    cv2.putText(frame, 'drone', (int(actual_position[0]), int(actual_position[1]) - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # plot trajectory
    cv2.line(frame, (
        mid_point(actual_position[0], actual_position[2]), # x1
        mid_point(actual_position[1], actual_position[3]) # y1
    ), (
        int(predicted_position[0] - ((actual_position[0] - actual_position[2]) / 2)), # x2
        int(predicted_position[1] - ((actual_position[1] - actual_position[3]) / 2)) # y2
    ), (0, 255, 0), 2)

    # add the current position to the list of positions
    drone_positions.append((mid_point(actual_position[0], actual_position[2]), mid_point(actual_position[1], actual_position[3])))

    # draw the line of the drone's path
    for i in range(1, len(drone_positions)):
        cv2.line(frame, drone_positions[i-1], drone_positions[i], (128, 0, 255), 2)

def detect_drone(frame, model, device='cpu'):
    # Convert the frame to an image
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Use the model to predict the drone's position
    results = model.predict(img, device=device)
    # Check if the drone is detected
    if len(results) > 0 and len(results[0].boxes) > 0:
        # Get the position of the drone
        drone_position = results[0].boxes.xyxy[0].cpu().numpy()
        return drone_position
    else:
        return None


