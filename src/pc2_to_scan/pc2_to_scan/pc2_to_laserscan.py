#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, LaserScan
import numpy as np
from sensor_msgs_py import point_cloud2 as pc2
import math
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
qos = QoSProfile(
    depth=10,
    reliability=QoSReliabilityPolicy.BEST_EFFORT
)
class Pc2ToLaserScan(Node):
    def __init__(self):
        super().__init__('pc2_to_laserscan')

        # ---- 사용자 환경에 맞춘 파라미터 ----
        self.declare_parameter('pc_topic', '/ground_segmentation/lidar')  
        self.declare_parameter('scan_topic', '/scan')          
        self.declare_parameter('angle_min', -3 * math.pi / 4)                    
        self.declare_parameter('angle_max', 3 * math.pi / 4)                      
        self.declare_parameter('angle_increment_deg', 0.25)               
        self.declare_parameter('range_min', 0.1)                          
        self.declare_parameter('range_max', 7.0)                         
        self.declare_parameter('use_closest_point', True)                 

        pc_topic = self.get_parameter('pc_topic').value
        scan_topic = self.get_parameter('scan_topic').value
        self.angle_min = self.get_parameter('angle_min').value
        self.angle_max = self.get_parameter('angle_max').value
        ang_inc_deg = self.get_parameter('angle_increment_deg').value
        self.angle_increment = math.radians(ang_inc_deg)
        self.range_min = self.get_parameter('range_min').value
        self.range_max = self.get_parameter('range_max').value
        self.use_closest_point = self.get_parameter('use_closest_point').value

        self.num_bins = int(round((self.angle_max - self.angle_min) / self.angle_increment))
        if self.num_bins <= 0:
            self.get_logger().error('Invalid angle range / increment.')
            raise RuntimeError('Invalid angle range')

        self.sub = self.create_subscription(PointCloud2, pc_topic, self.pc_callback, qos_profile_sensor_data)
        self.pub = self.create_publisher(LaserScan, scan_topic, qos_profile_sensor_data)
        self.get_logger().info(f'pc2->scan node ready, bins={self.num_bins}')

    def pc_callback(self, msg: PointCloud2):
        ranges = np.full(self.num_bins, float(self.range_max), dtype=np.float32)

        points = pc2.read_points_list(msg, field_names=('x','y','z'), skip_nans=True)

        for p in points:
            x, y, z = p.x, p.y, p.z
            r = math.hypot(x, y)
            if r < self.range_min or r > self.range_max:
                continue
            angle = math.atan2(y, x)  # -pi..pi
            idx_f = (angle - self.angle_min) / self.angle_increment
            idx = int(round(idx_f))
            if idx < 0 or idx >= self.num_bins:
                continue

            if self.use_closest_point:
                if r < ranges[idx]:
                    ranges[idx] = r
            else:
                ranges[idx] = min(ranges[idx], r)

        ranges_list = [float(r) for r in ranges]

        scan = LaserScan()
        scan.header = msg.header
        scan.angle_min = float(self.angle_min)
        scan.angle_max = float(self.angle_max)
        scan.angle_increment = float(self.angle_increment)
        scan.time_increment = 0.0
        scan.scan_time = 0.1   
        scan.range_min = float(self.range_min)
        scan.range_max = float(self.range_max)
        scan.ranges = ranges_list

        self.pub.publish(scan)

def main(args=None):
    rclpy.init(args=args)
    node = Pc2ToLaserScan()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()