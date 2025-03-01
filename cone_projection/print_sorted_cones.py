import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np

class ConeSubscriber(Node):
    def __init__(self):
        super().__init__('cone_subscriber')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/sorted_cones',
            self.cone_callback,
            10)
        self.subscription  # prevent unused variable warning

    def cone_callback(self, msg: Float32MultiArray):
        # 메시지 레이아웃 정보 가져오기
        num_cones = msg.layout.dim[0].size  # 콘 개수
        num_coords = msg.layout.dim[1].size  # 좌표 개수 (2: x, y)
        offset = msg.layout.data_offset

        # 데이터 변환
        try:
            data = np.array(msg.data[offset:])
            cones = data.reshape((num_cones, num_coords))

            self.get_logger().info(f"Received {num_cones} cones:")
            for i, (x, y) in enumerate(cones):
                self.get_logger().info(f"  Cone {i + 1}: (x={x:.3f}, y={y:.3f})")

        except Exception as e:
            self.get_logger().error(f"Data reshape error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ConeSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
