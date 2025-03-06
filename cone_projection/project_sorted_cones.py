#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
import time

import cv2
import numpy as np
import yaml
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from custom_interface.msg import ModifiedFloat32MultiArray
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image

from cone_projection.read_yaml import extract_configuration

# extrinsic 행렬 로드 함수
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

# 카메라 캘리브레이션 데이터 로드 함수
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

# sorted_cones 메시지 파싱 함수
def parse_sorted_cones(msg: ModifiedFloat32MultiArray) -> np.ndarray:
    data = np.array(msg.data, dtype=np.float64)
    if data.size % 2 != 0:
        raise ValueError("Sorted cones data size is not even.")
    markers = data.reshape((-1, 2))
    return markers

class SortedConesProjectionNode(Node):
    def __init__(self):
        super().__init__('sorted_cones_projection_node')
        
        # YAML 설정 파일 로드
        config_file = extract_configuration()
        if config_file is None:
            self.get_logger().error("Failed to extract configuration file.")
            return
        
        # extrinsic 및 intrinsic 데이터 로드
        config_folder = config_file['general']['config_folder']
        extrinsic_yaml = os.path.join(config_folder, config_file['general']['camera_extrinsic_calibration'])
        self.T_lidar_to_cam = load_extrinsic_matrix(extrinsic_yaml)
        
        camera_yaml = os.path.join(config_folder, config_file['general']['camera_intrinsic_calibration'])
        self.camera_matrix, self.dist_coeffs = load_camera_calibration(camera_yaml)
        
        self.get_logger().info("Loaded extrinsic:\n{}".format(self.T_lidar_to_cam))
        self.get_logger().info("Camera matrix:\n{}".format(self.camera_matrix))
        self.get_logger().info("Distortion coefficients:\n{}".format(self.dist_coeffs))
        
        # 구독 토픽 설정
        image_topic = config_file['camera']['image_topic']
        sorted_cones_topic = config_file['cones']['sorted_cones_topic']
        
        self.get_logger().info(f"Subscribing to image topic: {image_topic}")
        self.get_logger().info(f"Subscribing to sorted cones topic: {sorted_cones_topic}")
        
        self.image_sub = Subscriber(self, Image, image_topic)
        self.cones_sub = Subscriber(self, ModifiedFloat32MultiArray, sorted_cones_topic)
        
        # ApproximateTimeSynchronizer로 두 토픽 동기화
        self.ts = ApproximateTimeSynchronizer(
            [self.image_sub, self.cones_sub],
            queue_size=5,
            slop=0.1,
            allow_headerless=False,
        )
        self.ts.registerCallback(self.sync_callback)
        
        # 퍼블리시 토픽 설정
        projected_topic = config_file['camera']['projected_topic']
        self.pub_image = self.create_publisher(Image, projected_topic, 1)
        self.bridge = CvBridge()

        # 디버깅용 카운터 및 타이머
        self.image_count = 0
        self.cones_count = 0
        self.last_log_time = time.time()
        self.create_timer(1.0, self.log_message_counts)  # 1초마다 로그 출력

    # 1초마다 수신된 메시지 개수 출력
    def log_message_counts(self):
        current_time = time.time()
        if current_time - self.last_log_time >= 1.0:
            self.get_logger().info(f"Received {self.image_count} images and {self.cones_count} cones messages in the last second.")
            self.image_count = 0
            self.cones_count = 0
            self.last_log_time = current_time

    # 동기화된 이미지와 cones 데이터 처리
    def sync_callback(self, image_msg: Image, cones_msg: ModifiedFloat32MultiArray):
        self.image_count += 1
        self.cones_count += 1
        
        self.get_logger().info("Received synchronized image and cones data.")
        
        # 이미지 변환
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image message: {e}")
            return
        
        # 콘 데이터 파싱
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
        
        # 동차 좌표 변환 및 투사 좌표 계산
        markers_h = np.hstack((markers, np.zeros((n_markers, 1), dtype=np.float64), 
                            np.ones((n_markers, 1), dtype=np.float64)))
        markers_cam_h = markers_h @ self.T_lidar_to_cam.T
        markers_cam = markers_cam_h[:, :3].reshape(-1, 1, 3)
        
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)
        image_points, _ = cv2.projectPoints(markers_cam, rvec, tvec,
                                            self.camera_matrix, self.dist_coeffs)
        image_points = image_points.reshape(-1, 2)
        
        # 시각화 함수 호출
        self.visualize_points(cv_image, image_points)
        
        # 퍼블리시
        out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        out_msg.header = image_msg.header
        self.pub_image.publish(out_msg)
        self.get_logger().info(f"Published projected image with {n_markers} markers.")

    def visualize_points(self, cv_image: np.ndarray, image_points: np.ndarray):
        """
        이미지에 투사된 좌표를 시각화하는 함수
        :param cv_image: OpenCV 이미지 (BGR 형식)
        :param image_points: 투사된 픽셀 좌표 배열 (N, 2)
        """
        h, w = cv_image.shape[:2]
        self.get_logger().info(f"Image size: {h}x{w}")
        self.get_logger().info(f"Visualizing {len(image_points)} points.")
        self.get_logger().info(f"Image points: {image_points}")

        for i, (u, v) in enumerate(image_points):
            u_int = int(round(u))
            v_int = int(round(v))
            self.get_logger().info(f"Received Point {i}: ({u_int}, {v_int})")
            if 0 <= u_int < w and 0 <= v_int < h:
                # 원 그리기 (빨간색, 반지름 4)
                cv2.circle(cv_image, (u_int, v_int), 20, (0, 0, 255), -1)
                
                # 인덱스 라벨 추가 (초록색, 크기 0.5)
                label = f"ID:{i}"
                cv2.putText(cv_image, label, (u_int + 5, v_int - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                
                # 로그 출력 (디버깅용)
                self.get_logger().debug(f"Projected Point {i}: ({u_int}, {v_int})")

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