# Copyright 2023 michael. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import rclpy
from rclpy.node import Node
from pygame import *
import pygame
from roar_simulation_msgs.msg import VehicleControl
from std_msgs.msg import Header
from sensor_msgs.msg import Image

from typing import Optional
from rclpy.logging import LoggingSeverity
import numpy as np
from cv_bridge import CvBridge
import cv2
import threading
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image as PILImage
import matplotlib.pyplot as plt
from datetime import datetime

from pydantic import BaseModel


class State(BaseModel):
    target_speed: float = 0.0
    steering_angle: float = 0.0
    brake: float = 0.0
    reverse: bool = False

    def __repr__(self) -> str:
        return f"target_spd: {self.target_speed:.3f} steering_angle: {self.steering_angle:.3f} brake: {self.brake:.3f} reverse: {self.reverse}"


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.name = 'SimpleNet'
        
        self.conv1 = nn.Conv2d(3, 24, 5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, 5, stride=2)
        
        self.fc1 = nn.Linear(36*61*61, 100)
        self.fc2 = nn.Linear(100, 50)
        self.out = nn.Linear(50, 1)
        
        self.dropout = nn.Dropout(p = 0.1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x
    
class Net(nn.Module):
    def __init__ (self):
        super(Net, self).__init__()
        self.name = 'Net'

        self.conv1 = nn.Conv2d(3, 24, 5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, 5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, 5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, 3)
        self.conv5 = nn.Conv2d(64, 64, 3)
        
        self.fc1 = nn.Linear(64*25*25, 100) 
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.out = nn.Linear(10, 1)
        
        self.dropout = nn.Dropout(p = 0.1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.out(x)

        return x
    

class ManualControllerNode(Node):
    def __init__(self):
        super().__init__("roar_carla_manual_controller_node")

        # timers
        self.declare_parameter("refresh_rate", 0.05)
        self.declare_parameter("debug", False)
        self.declare_parameter("speed_increment", 0.1)
        self.declare_parameter("angle_increment", 1.0)
        self.declare_parameter("frame_id", "ego_vehicle")
        self.timer_callback = self.create_timer(
            self.get_parameter("refresh_rate").get_parameter_value().double_value,
            self.on_timer_callback,
        )

        # pub / sub
        self.cmd_publisher = self.create_publisher(VehicleControl, "roar_carla_cmd", 10)

        self.rgb_sub = self.create_subscription(Image, "image", lambda msg: self.on_image_received(msg, "image"), 10)
        self.l_rgb_sub = self.create_subscription(Image, "l_image", lambda msg: self.on_image_received(msg, "l_image"), 10)
        self.r_rgb_sub = self.create_subscription(Image, "r_image", lambda msg: self.on_image_received(msg, "r_image"), 10)

        self.cv_bridge_ = CvBridge()
        self.image: Optional[np.ndarray] = None
        self.image_l: Optional[np.ndarray] = None
        self.image_r: Optional[np.ndarray] = None
        self.image_to_save: Optional[np.ndarray] = None


        # Define the directory only once during the initialization
        current_date = datetime.now().strftime("%H%M%S%d%m%y")
        self.dataset_directory = f"./datasets/dataset_{current_date}/images"

        # JSON file path
        self.json_file_path = f"./datasets/dataset_{current_date}/data.json"

        if not os.path.exists(self.dataset_directory):
            os.makedirs(self.dataset_directory)
        
        if not os.path.exists(self.json_file_path):
            with open(self.json_file_path, 'w') as json_file:
                pass


        # pygame
        pygame.init()
        self.display = None
        self.surface = None
        pygame.display.set_caption(self.get_name())
        self.font = pygame.font.Font(None, 36)  # Choose the font and size

        # Select the CNN model 
        self.model = Net().to("cuda")
        model_path = "/home/roar/Self_Driving_ROAR/models/Net_0203_1853.pth"
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # Change this depending on the model/dataset
        self.labels_min = -8.5
        self.labels_max = 11.75

        # Steering angle predicted by the CNN
        self.steering_angle_predicted = 0.0

        # Transformations to apply to the images
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Self driving
        self.autonomous_controlling = False

        # Image counter
        self.image_counter = 0

        # Recording dataset
        self.recording_dataset = False      

        # misc
        self.state = State()
        self.speed_inc = (
            self.get_parameter("speed_increment").get_parameter_value().double_value
        )
        self.angle_inc = (
            self.get_parameter("angle_increment").get_parameter_value().double_value
        )

        if self.get_parameter("debug").get_parameter_value().bool_value:
            self.get_logger().set_level(LoggingSeverity.DEBUG)
        else:
            self.get_logger().set_level(LoggingSeverity.INFO)

    
    def on_image_received(self, msg: Image, topic_name: str):
        try:
            image = cv2.rotate(
                self.cv_bridge_.imgmsg_to_cv2(msg, desired_encoding="bgr8"),
                cv2.ROTATE_90_COUNTERCLOCKWISE,
            )

            # Specify image orientation
            if topic_name == "image":
                image_orientation = "front"
                self.image = cv2.flip(image, 0)
                self.image_to_save = self.image
            elif topic_name == "l_image":
                image_orientation = "left"
                self.image_l = cv2.flip(image, 0)
                self.image_to_save = self.image_l
            elif topic_name == "r_image":
                image_orientation = "right"
                self.image_r = cv2.flip(image, 0)
                self.image_to_save = self.image_r

            if image_orientation == "front":
                # Rotate the image
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                # Apply the same transformations as in the training
                image = self.transform(image)
                # Add a dimension for the batch
                image = image.unsqueeze(0).to("cuda")

                # Predict the steering angle
                with torch.no_grad():
                    steering_angle_out = self.model(image).item()
                
                # Denormalize the steering angle
                self.steering_angle_predicted = steering_angle_out * (self.labels_max - self.labels_min) + self.labels_min
            
                if self.autonomous_controlling == True:
                    # Update the steering angle command
                    self.state.steering_angle = self.steering_angle_predicted 
                    # log steering angle predicted
                    self.get_logger().info(f"Steering angle predicted: {self.steering_angle_predicted:.3f}")

            if self.recording_dataset == True:
                # Start a new thread to save the image and JSON data
                threading.Thread(target=self.save_image_and_json, args=(self.image_to_save, self.state, self.image_counter, image_orientation)).start()

                # Increment the image counter
                self.image_counter += 1

        except Exception as e:
            self.get_logger().error("Failed to convert image: %s" % str(e))
            return

    def save_image_and_json(self, image, state, image_counter, image_orientation):        
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image = cv2.flip(image, 1)

        # Save the image
        image_path = f"{self.dataset_directory}/image_{image_orientation}_{image_counter}.jpg"
        cv2.imwrite(image_path, image)

        # Modify steering angle based on orientation
        if image_orientation == "front":
            steering_angle_recorded = state.steering_angle
        elif image_orientation == "left":
            steering_angle_recorded = state.steering_angle + 2.5
        elif image_orientation == "right":
            steering_angle_recorded = state.steering_angle - 2.5

        # Save the steering angle and image path in a json file
        with open(self.json_file_path, 'a') as labels_file:
            labels_file.write('{"image": "'+image_path+'", "label": ' + str(steering_angle_recorded) + '}\n')


    def on_timer_callback(self):
        if self.image is not None and self.image_r is not None and self.image_l is not None:
            #stacked_image = np.vstack((self.image_l, self.image, self.image_r))
            #resized_stacked_image = cv2.resize(stacked_image, (0, 0), fx=1/2, fy=1/2)
            resized_stacked_image = self.image

            if self.surface is None or self.display is None:
                # First render
                self.display = pygame.display.set_mode(
                    (resized_stacked_image.shape[0], resized_stacked_image.shape[1]),
                    pygame.HWSURFACE | pygame.DOUBLEBUF,
                )
                self.surface = pygame.surfarray.make_surface(cv2.cvtColor(resized_stacked_image, cv2.COLOR_BGR2RGB))

            pygame_image = pygame.surfarray.make_surface(cv2.cvtColor(resized_stacked_image, cv2.COLOR_BGR2RGB))
            self.display.blit(pygame_image, (0, 0))

            text = self.font.render(
                self.state.__repr__(), True, (255, 0, 0)
            )  # Render the state as text
            self.display.blit(text, (10, 10))  # Blit the text onto the display surface

            # Write a O if recording
            if self.recording_dataset == True:
                text = self.font.render(
                    "o REC", True, (0, 255, 0)
                )
                self.display.blit(text, (10, 50))

            # Write an A if autonomous controlling
            if self.autonomous_controlling == True:
                text = self.font.render(
                    "Autonomous mode: ON", True, (0, 255, 0)
                )
                self.display.blit(text, (10, 70))
            else:
                text = self.font.render(
                    "Autonomous mode: OFF", True, (255, 0, 0)
                )
                self.display.blit(text, (10, 70))

            # Write the steering angle predicted by the CNN
            text = self.font.render(
                f"Steering angle predicted: {self.steering_angle_predicted:.3f}", True, (0, 255, 0)
            )
            self.display.blit(text, (10, 90))

            
            pygame.display.flip()

        self.p_parse_event()
        self.p_publish(self.state)

    def p_publish(self, state: State):
        header: Header = Header()
        header.frame_id = (
            self.get_parameter("frame_id").get_parameter_value().string_value
        )

        header.stamp = self.get_clock().now().to_msg()

        control: VehicleControl = VehicleControl()
        control.header = header
        control.target_speed = float(state.target_speed)
        control.steering_angle = float(state.steering_angle)
        control.brake = float(state.brake)
        control.reverse = bool(state.reverse)
        self.cmd_publisher.publish(control)

    def p_parse_event(self):
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_r:
                    self.state.reverse = not self.state.reverse

                if event.key == K_w or event.key == K_UP:
                    self.state.target_speed = self.state.target_speed + self.speed_inc
                if event.key == K_s or event.key == K_DOWN:
                    self.state.target_speed = self.state.target_speed - self.speed_inc

                if event.key == K_d or event.key == K_RIGHT:
                    self.state.steering_angle = (
                        self.state.steering_angle + self.angle_inc
                    )
                if event.key == K_a or event.key == K_LEFT:
                    self.state.steering_angle = (
                        self.state.steering_angle - self.angle_inc
                    )
                if event.key == K_o:
                    self.recording_dataset = True
                if event.key == K_p:
                    self.recording_dataset = False
                if event.key == K_i:
                    self.autonomous_controlling = True
                if event.key == K_u and self.autonomous_controlling == True:
                    self.autonomous_controlling = False
                if event.key == K_SPACE:
                    self.state.brake = 1 - self.state.brake
        self.get_logger().debug(f"Control: {self.state.__repr__()}")

    def destroy_node(self):
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ManualControllerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
