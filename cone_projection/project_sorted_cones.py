#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node

import cv2
import numpy as np
import yaml
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from std_msgs.msg import Float32MultiArray, Image

from cone_projection.read_yaml import extract_configuration

def load_extrinsic_matrix(yaml_path: str) -> np.ndarray:
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"No extrinsic file found: {yaml_path}")
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    if 'extrinsic_matrix' not in data:
        raise KeyError(f"YAML {yaml_path} has no 'extrinsic_matrix' key.")
    matrix_list = data['extrinsic_matrix']
    T = np.array(matrix_list, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError("Extrinsic matrix is not 4x4.")
    return T

def load_camera_calibration(yaml_path: str) -> (np.ndarray, np.ndarray):
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"No camera calibration file: {yaml_path}")
    with open(yaml_path, 'r') as f:
        calib_data = yaml.safe_load(f)
    cam_mat_data = calib_data['camera_matrix']['data']
    camera_matrix = np.array(cam_mat_data, dtype=np.float64)
    dist_data = calib_data['distortion_coefficients']['data']
    dist_coeffs = np.array(dist_data, dtype=np.float64).reshape((1, -1))
    return camera_matrix, dist_coeffs

def parse_sorted_cones(msg: Float32MultiArray) -> np.ndarray:
    """
    sorted_cones 토픽 메시지에서 (x, y) 좌표 데이터를 파싱합니다.
    메시지의 data 필드는 1차원 배열로, 각 두 개의 연속된 값이 하나의 마커 좌표임을 가정합니다.
    """
    data = np.array(msg.data, dtype=np.float64)
    if data.size % 2 != 0:
        raise ValueError("Sorted cones data size is not even.")
    markers = data.reshape((-1, 2))
    return markers

class SortedConesProjectionNode(Node):
    def __init__(self):
        super().__init__('sorted_cones_projection_node')
        
        # 설정 파일 추출 (YAML 구성 파일)
        config_file = extract_configuration()
        if config_file is None:
            self.get_logger().error("Failed to extract configuration file.")
            return
        
        config_folder = config_file['general']['config_folder']
        extrinsic_yaml = os.path.join(config_folder, config_file['general']['camera_extrinsic_calibration'])
        self.T_lidar_to_cam = load_extrinsic_matrix(extrinsic_yaml)
        
        camera_yaml = os.path.join(config_folder, config_file['general']['camera_intrinsic_calibration'])
        self.camera_matrix, self.dist_coeffs = load_camera_calibration(camera_yaml)
        
        self.get_logger().info("Loaded extrinsic:\n{}".format(self.T_lidar_to_cam))
        self.get_logger().info("Camera matrix:\n{}".format(self.camera_matrix))
        self.get_logger().info("Distortion coefficients:\n{}".format(self.dist_coeffs))
        
        # 이미지와 sorted_cones 토픽 구독 (config 파일에 토픽 이름이 정의되어 있다고 가정)
        image_topic = config_file['camera']['image_topic']
        sorted_cones_topic = config_file['cones']['sorted_cones_topic']
        
        self.get_logger().info(f"Subscribing to image topic: {image_topic}")
        self.get_logger().info(f"Subscribing to sorted cones topic: {sorted_cones_topic}")
        
        self.image_sub = Subscriber(self, Image, image_topic)
        self.cones_sub = Subscriber(self, Float32MultiArray, sorted_cones_topic)
        
        self.ts = ApproximateTimeSynchronizer(
            [self.image_sub, self.cones_sub],
            queue_size=5,
            slop=0.07
        )
        self.ts.registerCallback(self.sync_callback)
        
        projected_topic = config_file['camera']['projected_topic']
        self.pub_image = self.create_publisher(Image, projected_topic, 1)
        self.bridge = CvBridge()
        
    def sync_callback(self, image_msg: Image, cones_msg: Float32MultiArray):
        # 1. 이미지 메시지를 OpenCV 이미지로 변환
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        
        # 2. sorted_cones 메시지에서 (x, y) 좌표 파싱
        try:
            markers = parse_sorted_cones(cones_msg)
        except Exception as e:
            self.get_logger().error(f"Error parsing sorted cones: {e}")
            markers = np.empty((0, 2), dtype=np.float64)
        
        n_markers = markers.shape[0]
        if n_markers == 0:
            self.get_logger().info("No sorted cones markers to project.")
            out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            out_msg.header = image_msg.header
            self.pub_image.publish(out_msg)
            return
        
        # 3. (x, y) 좌표를 4차원 동차좌표 ([x, y, 0, 1])로 변환
        markers_h = np.hstack((markers, np.zeros((n_markers, 1), dtype=np.float64), 
                                        np.ones((n_markers, 1), dtype=np.float64)))
        
        # 4. 동차좌표를 extrinsic 행렬로 변환하여 카메라 좌표계로 변환
        markers_cam_h = markers_h @ self.T_lidar_to_cam.T
        markers_cam = markers_cam_h[:, :3]
        
        # 5. 카메라 앞에 있는 포인트(z > 0)만 필터링
        mask_in_front = markers_cam[:, 2] > 0.0
        markers_cam_front = markers_cam[mask_in_front]
        if markers_cam_front.shape[0] == 0:
            self.get_logger().info("No markers in front of camera (z>0).")
            out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            out_msg.header = image_msg.header
            self.pub_image.publish(out_msg)
            return
        
        # 6. 3D 포인트를 2D 이미지 픽셀 좌표로 투영 (회전 및 변환 벡터는 0으로 설정)
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)
        image_points, _ = cv2.projectPoints(markers_cam_front, rvec, tvec,
                                            self.camera_matrix, self.dist_coeffs)
        image_points = image_points.reshape(-1, 2)
        
        # 7. 투영된 포인트들을 이미지에 시각화 (예: 빨간 원)
        h, w = cv_image.shape[:2]
        for (u, v) in image_points:
            u_int = int(round(u))
            v_int = int(round(v))
            if 0 <= u_int < w and 0 <= v_int < h:
                cv2.circle(cv_image, (u_int, v_int), 4, (0, 0, 255), -1)
        
        out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        out_msg.header = image_msg.header
        self.pub_image.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SortedConesProjectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
