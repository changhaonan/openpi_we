import numpy as np
import pyrealsense2 as rs
from openpi_client import websocket_client_policy
import logging
import argparse
import cv2
import pickle
import zmq
from typing import Dict
import time

class ZMQClientRobot():
    """A class representing a ZMQ client for a leader robot."""

    def __init__(self, port: int = 6001, host: str = "127.0.0.1"):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(f"tcp://{host}:{port}")

    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        request = {"method": "get_joint_state"}
        send_message = pickle.dumps(request)
        self._socket.send(send_message)
        result = pickle.loads(self._socket.recv())
        return result

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        """Command the leader robot to the given state.

        Args:
            joint_state (T): The state to command the leader robot to.
        """
        request = {
            "method": "command_joint_state",
            "args": {"joint_state": joint_state},
        }
        send_message = pickle.dumps(request)
        self._socket.send(send_message)
        result = pickle.loads(self._socket.recv())
        return result


def setup_realsense_camera(serial_number):
    """Setup a RealSense camera with the given serial number."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_number)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    pipeline.start(config)
    return pipeline


def get_camera_frame(pipeline):
    """Get a frame from the RealSense camera and resize it to 224x224."""
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    
    # Convert from BGR to RGB
    color_image = color_image[..., ::-1]
    
    # Resize to 224x224 and transpose to channel-first format (3, 224, 224)
    resized_image = np.array(cv2.resize(color_image, (320, 240)))
    resized_image = np.transpose(resized_image, (2, 0, 1))
    
    return resized_image


def main():
    parser = argparse.ArgumentParser(description='RealSense client for World Engine')
    parser.add_argument('--host', type=str, default='192.168.1.93', help='Server host')
    parser.add_argument('--port', type=int, default=8000, help='Server port')
    args = parser.parse_args()

    # Camera serial numbers
    CAM_HIGH = "218622278781"
    CAM_LEFT_WRIST = "218622279087"
    CAM_RIGHT_WRIST = "218622274594"

    # Setup cameras
    logging.info("Setting up cameras...")
    cam_high_pipeline = setup_realsense_camera(CAM_HIGH)
    cam_left_pipeline = setup_realsense_camera(CAM_LEFT_WRIST)
    cam_right_pipeline = setup_realsense_camera(CAM_RIGHT_WRIST)

    # Save test frames from each camera
    logging.info("Saving test frames...")
    test_frames = {
        "cam_high": get_camera_frame(cam_high_pipeline),
        "cam_left": get_camera_frame(cam_left_pipeline), 
        "cam_right": get_camera_frame(cam_right_pipeline)
    }

    # Save each frame
    for camera_name, frame in test_frames.items():
        # Convert from CHW back to HWC format for saving
        frame = np.transpose(frame, (1, 2, 0))
        # Convert from RGB back to BGR for cv2
        frame = frame[..., ::-1]
        cv2.imwrite(f"{camera_name}.jpg", frame)
    logging.info("Test frames saved")

    # Initialize ZMQ client
    logging.info("Initializing ZMQ client...")
    zmq_client = ZMQClientRobot()
    state = zmq_client.get_joint_state()
    logging.info(f"Joint state: {state}")

    # Initialize WebSocket client
    logging.info(f"Connecting to server at {args.host}:{args.port}...")
    policy = websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logging.info(f"Server metadata: {policy.get_server_metadata()}")

    try:
        while True:
            # Get frames from all cameras
            cam_high_frame = get_camera_frame(cam_high_pipeline)
            cam_left_frame = get_camera_frame(cam_left_pipeline)
            cam_right_frame = get_camera_frame(cam_right_pipeline)

            # Create observation dictionary
            obs = {
                "observation.state": zmq_client.get_joint_state(),
                "observation.images.cam_high": cam_high_frame,
                "observation.images.cam_left_wrist": cam_left_frame,
                "observation.images.cam_right_wrist": cam_right_frame,
                "action": np.zeros([50, 14], dtype=np.float32),
            }

            # Send observation to server and get response
            try:
                response = policy.infer(obs)
                print("Server response:", response)
                for joint_state in response['actions']:
                    zmq_client.command_joint_state(joint_state)
                    time.sleep(0.01)
            except Exception as e:
                logging.error(f"Error during inference: {e}")

    finally:
        # Stop all camera pipelines
        cam_high_pipeline.stop()
        cam_left_pipeline.stop()
        cam_right_pipeline.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 