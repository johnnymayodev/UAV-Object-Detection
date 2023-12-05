import os


# directories
MODEL_DIR = 'models/'
VIDEOS_DIR = 'videos/'
DATASET_DIR = 'datasets/Drone Dataset/'
HOME = os.getcwd()

# files
PATH_TO_YAML = DATASET_DIR + 'data.yaml'

# model settings
MODEL_NAME = 'drone_obj_detector'
IMG_SIZE = 640

if __name__ == "__main__":
    print("ERROR: This file is not meant to be run directly. Please run 'main.py' instead.")