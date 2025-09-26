#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from tier4_perception_msgs.msg import DetectedObjectsWithFeature
from f110_msgs.msg import ObstacleArray, Obstacle as ObstacleMessage
from f110_msgs.msg import Wpnt, WpntArray
import numpy as np
from tf_transformations import euler_from_quaternion

# Frenet 변환 클래스 (ForzaETH와 동일한 인터페이스 가정)
# converter.get_frenet(x_array, y_array) -> s_array, d_array
from frenet_conversion.frenet_converter import FrenetConverter 

class ClusterToObstacle(Node):
    def __init__(self):
        super().__init__('cluster_to_obstacle')

        # --- Parameters ---
        self.declare_parameter("frenet_waypoints_topic", "/global_waypoints")
        self.declare_parameter("cluster_topic", "/clusters")
        self.declare_parameter("obstacle_pub_topic", "/perception/detection/raw_obstacles")

        # --- Subscribers ---
        self.cluster_sub = self.create_subscription(
            DetectedObjectsWithFeature,
            self.get_parameter("cluster_topic").value,
            self.cluster_callback,
            10
        )

        # --- Frenet converter ---
        self.converter = None
        self.waypoints_sub = self.create_subscription(
            WpntArray,  # 여기를 올바른 메시지 타입으로 수정
            self.get_parameter("frenet_waypoints_topic").value,
            self.waypoints_callback,
            10
        )
        # --- Publisher ---
        self.obstacles_pub = self.create_publisher(
            ObstacleArray,
            self.get_parameter("obstacle_pub_topic").value,
            10
        )

    def waypoints_callback(self, msg):

        xs = [wpnt.x_m for wpnt in msg.wpnts]
        ys = [wpnt.y_m for wpnt in msg.wpnts]
        psis = [wpnt.psi_rad for wpnt in msg.wpnts]
        self.converter = FrenetConverter(np.array(xs), np.array(ys), np.array(psis))
        # self.get_logger().info("FrenetConverter initialized")

    def cluster_callback(self, msg: DetectedObjectsWithFeature):
        if self.converter is None:
            self.get_logger().warn("FrenetConverter not initialized yet")
            return

        obs_array_msg = ObstacleArray()
        obs_array_msg.header.stamp = self.get_clock().now().to_msg()
        obs_array_msg.header.frame_id = "map"

        x_centers = []
        y_centers = []

        for obj in msg.feature_objects:
            x = obj.object.kinematics.pose_with_covariance.pose.position.x
            y = obj.object.kinematics.pose_with_covariance.pose.position.y
            x_centers.append(x)
            y_centers.append(y)

        s_array, d_array = self.converter.get_frenet(np.array(x_centers), np.array(y_centers))

        for idx, obj in enumerate(msg.feature_objects):
            size = max(obj.object.shape.dimensions.x, obj.object.shape.dimensions.y)

            obs_msg = ObstacleMessage()
            obs_msg.id = idx
            obs_msg.s_center = s_array[idx]
            obs_msg.d_center = d_array[idx]
            obs_msg.size = size
            obs_msg.s_start = s_array[idx] - size/2
            obs_msg.s_end = s_array[idx] + size/2
            obs_msg.d_left = d_array[idx] + size/2
            obs_msg.d_right = d_array[idx] - size/2

            # Optional: yaw 저장 가능
            # obs_msg.theta = yaw  

            obs_array_msg.obstacles.append(obs_msg)

        self.obstacles_pub.publish(obs_array_msg)
        self.get_logger().info(f"Published {len(obs_array_msg.obstacles)} obstacles")

def main():
    rclpy.init()
    node = ClusterToObstacle()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()