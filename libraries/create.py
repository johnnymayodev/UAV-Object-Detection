import os
import shutil

import torch

from ultralytics import YOLO
from ultralytics import settings

import libraries.common as cv


def main(epochs=50, batch_size=16, learning_rate=0.001, optimizer="auto", device="cpu"):
    # changing ultralytics settings. this will fix errors for not being able to find the data and prevent tensorboard and wandb from running.
    settings.reset()
    settings.update({"tensorboard": False})
    settings.update({"wandb": False})

    print("\ncreating model...")
    model = YOLO("yolov8n.yaml")  # get a YOLOv8 model from ultralytics
    model = YOLO(cv.MODEL_DIR + "yolov8n.pt")  # saves the model to the model directory
    os.rename(cv.MODEL_DIR + "yolov8n.pt", cv.MODEL_DIR + "template_model.pt")
    model = YOLO(
        cv.MODEL_DIR + "template_model.pt"
    )  # load the model from the model directory

    if os.path.exists('runs/'):
        selection = input("WARNING: this will delete all previous training data. would you like to continue? (y/n)\n")
        if selection == "y":
            shutil.rmtree('runs/')
        else:
            print("exiting...")
            exit()

    print(
        f"\nrunning initial training with parameters:\n"
        "epochs: {epochs}\n"
        "batch size: {batch_size}\n"
        "learning rate: {learning_rate}\n"
        "optimizer: {optimizer}\ndevice: {device}\n"
    )
    model.train(
        data=cv.PATH_TO_YAML,
        epochs=epochs,
        batch=batch_size,
        lr0=learning_rate,
        lrf=learning_rate,
        device=device,
    )

    # saving the model
    print("\nsaving model...")

    if not os.path.exists("runs/"):
        raise FileNotFoundError("ERROR: 'runs/' directory not found. Please try again.")

    if os.path.exists(cv.MODEL_DIR + "fresh_model.pt"):
        selection = input("WARNING: this will delete all previous training data. would you like to continue? (y/n)\n")
        if selection == "y":
            os.remove(cv.MODEL_DIR + "fresh_model.pt")
            shutil.copy2('runs/detect/train/weights/best.pt', cv.MODEL_DIR + 'fresh_model.pt')
        else:
            print("exiting...")
            exit()

    else:
        shutil.copy2('runs/detect/train/weights/best.pt', cv.MODEL_DIR + 'fresh_model.pt')

    print("\nmodel saved successfully.")

    shutil.rmtree("runs/")


if __name__ == "__main__":
    print(
        "ERROR: This file is not meant to be run directly. Please run 'main.py' instead."
    )
