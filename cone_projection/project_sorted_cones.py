import os
import rclpy
from rclpy.node import Node

import cv2
import numpy as np
import yaml
import struct

from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from cone_projection.read_yaml import extract_configuration


def load_extrinsic_matrix(yaml_path: str) -> np.ndarray:
    if not os.path.isfile(yaml_path):  # 파일 존재 여부 확인
        raise FileNotFoundError(f"No extrinsic file found: {yaml_path}")

    with open(yaml_path, 'r') as f:  # YAML 파일 열기
        data = yaml.safe_load(f)  # YAML 데이터 로드

    if 'extrinsic_matrix' not in data:  # 'extrinsic_matrix' 키 존재 여부 확인
        raise KeyError(f"YAML {yaml_path} has no 'extrinsic_matrix' key.")

    matrix_list = data['extrinsic_matrix']  # 외부 행렬 데이터 가져오기
    T = np.array(matrix_list, dtype=np.float64)  # numpy 배열로 변환
    if T.shape != (4, 4):  # 행렬 크기 확인
        raise ValueError("Extrinsic matrix is not 4x4.")
    return T

def load_camera_calibration(yaml_path: str) -> (np.ndarray, np.ndarray):
    if not os.path.isfile(yaml_path):  # 파일 존재 여부 확인
        raise FileNotFoundError(f"No camera calibration file: {yaml_path}")

    with open(yaml_path, 'r') as f:  # YAML 파일 열기
        calib_data = yaml.safe_load(f)  # YAML 데이터 로드

    cam_mat_data = calib_data['camera_matrix']['data']  # 카메라 행렬 데이터 가져오기
    camera_matrix = np.array(cam_mat_data, dtype=np.float64)  # numpy 배열로 변환

    dist_data = calib_data['distortion_coefficients']['data']  # 왜곡 계수 데이터 가져오기
    dist_coeffs = np.array(dist_data, dtype=np.float64).reshape((1, -1))  # numpy 배열로 변환 및 재구성

    return camera_matrix, dist_coeffs

def parse_sorted_cones()
    # /sorted_cones 토픽을 구독받아서 들어온 토픽을 파싱하고 (x,y) 좌표 데이터를 뽑아냄
    # 뽑아낸 좌표 데이터를 배열 형태로 반환
    return world_coordinates

class LidarCameraProjectionNode(Node):  
    def __init__(self):
        super().__init__('lidar_camera_projection_node') 
        
        config_file = extract_configuration() 
        if config_file is None:  
            self.get_logger().error("Failed to extract configuration file.")
            return
        
        best_effort_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        config_folder = config_file['general']['config_folder']
        extrinsic_yaml = config_file['general']['camera_extrinsic_calibration']
        extrinsic_yaml = os.path.join(config_folder, extrinsic_yaml)
        self.T_lidar_to_cam = load_extrinsic_matrix(extrinsic_yaml)

        camera_yaml = config_file['general']['camera_intrinsic_calibration']
        camera_yaml = os.path.join(config_folder, camera_yaml)
        self.camera_matrix, self.dist_coeffs = load_camera_calibration(camera_yaml)

        # 로드된 행렬 및 보정 데이터 로그 출력
        self.get_logger().info("Loaded extrinsic:\n{}".format(self.T_lidar_to_cam))
        self.get_logger().info("Camera matrix:\n{}".format(self.camera_matrix))
        self.get_logger().info("Distortion coeffs:\n{}".format(self.dist_coeffs))

        # 라이다 및 이미지 토픽 구독 설정
        lidar_topic = config_file['lidar']['lidar_topic']
        image_topic = config_file['camera']['image_topic']
        self.get_logger().info(f"Subscribing to lidar topic: {lidar_topic}")
        self.get_logger().info(f"Subscribing to image topic: {image_topic}")

        self.image_sub = Subscriber(self, Image, image_topic)
        self.lidar_sub = Subscriber(self, PointCloud2, lidar_topic, qos_profile=best_effort_qos)

        # 메시지 동기화 설정
        self.ts = ApproximateTimeSynchronizer( # 두 개 이상의 ROS 메시지 토픽을 동기화, 타임스탬프가 가장 가까운 메시지를 동기화
            [self.image_sub, self.lidar_sub], # 구독자 리스트
            queue_size=5,
            slop=0.07 # 타임스탬프 차이 허용 시간
        )
        self.ts.registerCallback(self.sync_callback)  # 동기화되면 호출될 콜백(sync_callback) 등록, 동기화된 메시지 쌍을 sync_callback에 전달

        # 투영된 이미지 토픽 퍼블리셔 생성
        projected_topic = config_file['camera']['projected_topic'] # 투영된 이미지 토픽 이름
        self.pub_image = self.create_publisher(Image, projected_topic, 1) # 투영된 이미지 토픽 퍼블리셔 생성
        self.bridge = CvBridge()  # CvBridge 객체 생성

        self.skip_rate = 1  # 샘플링 비율 설정

    def sync_callback(self, image_msg: Image, lidar_msg: PointCloud2): 
        # 1. ROS 이미지 메시지를 OpenCV 이미지로 변환
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')  # ROS Image -> OpenCV BGR 이미지

        # 2. 포인트 클라우드 메시지를 XYZ 배열로 변환
        xyz_lidar = pointcloud2_to_xyz_array_fast(lidar_msg, skip_rate=self.skip_rate)  # LiDAR 데이터 -> (N, 3) XYZ 배열
        n_points = xyz_lidar.shape[0]
        if n_points == 0:  # LiDAR 데이터가 비어 있을 경우 처리
            self.get_logger().warn("Empty cloud. Nothing to project.")
            # 처리 없이 원본 이미지를 퍼블리시
            out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')  # OpenCV 이미지 -> ROS Image 메시지
            out_msg.header = image_msg.header
            self.pub_image.publish(out_msg)  # 퍼블리시
            return

        # 3. 포인트 클라우드 데이터 준비: 동차 좌표 변환
        xyz_lidar_f64 = xyz_lidar.astype(np.float64)  # LiDAR 데이터를 float64 형식으로 변환
        ones = np.ones((n_points, 1), dtype=np.float64)  # 동차 좌표 계산을 위한 1 추가
        xyz_lidar_h = np.hstack((xyz_lidar_f64, ones))  # 동차 좌표로 변환 (N, 4)

        # 4. LiDAR 좌표계를 카메라 좌표계로 변환
        xyz_cam_h = xyz_lidar_h @ self.T_lidar_to_cam.T  # 변환 행렬(T) 적용: LiDAR -> 카메라 좌표계
        xyz_cam = xyz_cam_h[:, :3]  # 동차 좌표에서 3D 좌표(x, y, z) 추출

        # 5. 카메라 앞에 있는 포인트 필터링
        mask_in_front = (xyz_cam[:, 2] > 0.0)  # z > 0: 카메라 앞에 위치한 포인트만 선택
        xyz_cam_front = xyz_cam[mask_in_front]  # 필터링된 3D 포인트
        n_front = xyz_cam_front.shape[0]
        if n_front == 0:  # 카메라 앞에 포인트가 없을 경우 처리
            self.get_logger().info("No points in front of camera (z>0).")
            # 처리 없이 원본 이미지를 퍼블리시
            out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')  # OpenCV 이미지 -> ROS Image 메시지
            out_msg.header = image_msg.header
            self.pub_image.publish(out_msg)  # 퍼블리시
            return

        # 6. 3D 포인트를 2D 이미지로 투영
        rvec = np.zeros((3,1), dtype=np.float64)  # 카메라 회전 벡터 (0 초기화)
        tvec = np.zeros((3,1), dtype=np.float64)  # 카메라 변환 벡터 (0 초기화)
        image_points, _ = cv2.projectPoints(
            xyz_cam_front,  # 3D 포인트 (카메라 좌표계)
            rvec, tvec,     # 회전 및 이동 벡터
            self.camera_matrix,  # 카메라 매트릭스
            self.dist_coeffs      # 렌즈 왜곡 계수
        )
        image_points = image_points.reshape(-1, 2)  # 결과: (N, 2) 형태의 2D 이미지 포인트 배열

        # 7. 투영된 2D 포인트를 이미지 위에 시각화
        h, w = cv_image.shape[:2]  # 이미지 크기 가져오기 (높이, 너비)
        for (u, v) in image_points:  # 각 포인트에 대해
            u_int = int(u + 0.5)  # 정수형 좌표로 변환 (반올림)
            v_int = int(v + 0.5)
            if 0 <= u_int < w and 0 <= v_int < h:  # 포인트가 이미지 범위 내에 있는 경우
                cv2.circle(cv_image, (u_int, v_int), 2, (0, 255, 0), -1)  # 초록색 원으로 포인트 표시

        # 8. 처리 결과 이미지를 ROS 메시지로 변환 및 퍼블리시
        out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')  # OpenCV 이미지 -> ROS Image 메시지
        out_msg.header = image_msg.header
        self.pub_image.publish(out_msg)  # 퍼블리시

def main(args=None):
    rclpy.init(args=args)
    node = LidarCameraProjectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()