import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import cv2

class ConeVisualizer(Node):
    def __init__(self):
        super().__init__('cone_visualizer')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/sorted_cones',
            self.cone_callback,
            10)
        self.subscription  # unused variable 방지

        # OpenCV 창 설정
        self.window_name = "Cone Visualization"
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

        self.image_width = 800
        # 변경: scale을 50 픽셀/미터로 설정하여 더 넓은 영역을 볼 수 있도록 함.
        self.scale = 50  
        self.margin = 50  # 하단 마진
        # 변경: 최소 x축 양의 방향 12m(12*50=600픽셀)를 포함하고, 여유를 주어 이미지 높이를 670픽셀로 설정
        self.image_height = int(12 * self.scale + self.margin + 20)  
        self.origin = (self.image_width // 2, self.image_height - self.margin)

    def cone_callback(self, msg: Float32MultiArray):
        # layout.dim[0].size: 클러스터 개수, layout.dim[1].size: 클러스터 당 좌표 개수 (여기서는 2: x, y)
        num_cones = msg.layout.dim[0].size
        num_coords = msg.layout.dim[1].size
        offset = msg.layout.data_offset

        #self.get_logger().info(f"Received message: {msg}")
        #self.get_logger().info(f"num_cones: {num_cones}, num_coords: {num_coords}, offset: {offset}")

        try:
            data = np.array(msg.data[offset:])
            #self.get_logger().info(f"Raw data: {data}")

            cones = data.reshape((num_cones, num_coords))
            #self.get_logger().info(f"Reshaped data: {cones}")

        except Exception as e:
            self.get_logger().error(f"Data reshape error: {e}")
            return

        # 흰색 배경 이미지 생성
        img = np.full((self.image_height, self.image_width, 3), 255, dtype=np.uint8)

        # grid를 그리기 위한 world 좌표 범위 설정
        # x: 전방 방향, y: 좌우(양의 방향이 왼쪽)
        x_min, x_max = -2, 12
        y_min, y_max = -8, 8

        # world 좌표 -> 이미지 좌표 변환 함수
        def world_to_image(x, y):
            # x: 전방, y: 좌우 (양의 방향이 왼쪽)
            u = int(self.origin[0] - y * self.scale)  # y positive → 왼쪽
            v = int(self.origin[1] - x * self.scale)  # x positive → 위쪽 (전방)
            return (u, v)

        # 1m 간격의 horizontal grid line (x 고정) 그리기
        for x_val in np.arange(x_min, x_max + 1, 1):
            pt1 = world_to_image(x_val, y_min)
            pt2 = world_to_image(x_val, y_max)
            cv2.line(img, pt1, pt2, (220, 220, 220), 1)  # 연한 회색 선
            # x 좌표 레이블 (왼쪽 하단)
            label_pt = world_to_image(x_val, y_min)
            cv2.putText(img, f"{x_val}m", (label_pt[0] + 5, label_pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        # 1m 간격의 vertical grid line (y 고정) 그리기
        for y_val in np.arange(y_min, y_max + 1, 1):
            pt1 = world_to_image(x_min, y_val)
            pt2 = world_to_image(x_max, y_val)
            cv2.line(img, pt1, pt2, (220, 220, 220), 1)
            # y 좌표 레이블 (좌측 상단)
            label_pt = world_to_image(x_min, y_val)
            cv2.putText(img, f"{y_val}m", (label_pt[0] + 5, label_pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        # 상단 왼쪽에 cone 개수 표시
        cone_count_text = f"Cones: {num_cones}"
        cv2.putText(img, cone_count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

        # 내 차량 원점에 좌표 축 그리기 (파란색 화살표)
        axis_length = 50  # 화살표 길이 (픽셀)
        # x축: 전방 (이미지에서 위쪽 방향)
        cv2.arrowedLine(img, self.origin, (self.origin[0], self.origin[1] - axis_length),
                        (255, 0, 0), 2, tipLength=0.3)
        cv2.putText(img, 'x', (self.origin[0] - 15, self.origin[1] - axis_length - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        # y축: 양의 방향이 왼쪽이므로, 원점에서 왼쪽으로 화살표
        cv2.arrowedLine(img, self.origin, (self.origin[0] - axis_length, self.origin[1]),
                        (255, 0, 0), 2, tipLength=0.3)
        cv2.putText(img, 'y', (self.origin[0] - axis_length - 15, self.origin[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        # cone들을 이미지 좌표로 변환하여 그리기
        for cone in cones:
            x, y = cone
            u, v = world_to_image(x, y)
            #self.get_logger().info(f"Cone at world: ({x}, {y}), image: ({u}, {v}))")

            cv2.circle(img, (u, v), 5, (0, 255, 0), -1)
            text = f"({x:.2f}, {y:.2f})"
            cv2.putText(img, text, (u + 5, v - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

        # 내 차량(원점) 표시 (빨간색 원)
        cv2.circle(img, self.origin, 8, (0, 0, 255), -1)

        cv2.imshow(self.window_name, img)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ConeVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
