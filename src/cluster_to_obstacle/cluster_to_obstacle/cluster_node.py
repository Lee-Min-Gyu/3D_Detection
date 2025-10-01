#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from tier4_perception_msgs.msg import DetectedObjectsWithFeature
from f110_msgs.msg import ObstacleArray, Obstacle as ObstacleMessage
from f110_msgs.msg import WpntArray
import numpy as np
from tf_transformations import euler_from_quaternion

# TF2 imports
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PointStamped
import tf2_geometry_msgs.tf2_geometry_msgs  # PointStamped 변환용

# Frenet 변환 클래스
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

        self.converter = None
        self.waypoints_sub = self.create_subscription(
            WpntArray,
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

        # --- TF2 buffer & listener ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def waypoints_callback(self, msg):
        xs = [wpnt.x_m for wpnt in msg.wpnts]
        ys = [wpnt.y_m for wpnt in msg.wpnts]
        psis = [wpnt.psi_rad for wpnt in msg.wpnts]
        self.converter = FrenetConverter(np.array(xs), np.array(ys), np.array(psis))

    def cluster_callback(self, msg: DetectedObjectsWithFeature):
        if self.converter is None:
            self.get_logger().warn("FrenetConverter not initialized yet")
            return

        obs_array_msg = ObstacleArray()
        obs_array_msg.header.stamp = self.get_clock().now().to_msg()
        obs_array_msg.header.frame_id = "map"

        x_centers = []
        y_centers = []

        # --- TF2 변환: livox_frame -> map ---
        for obj in msg.feature_objects:
            point_livox = PointStamped()
            point_livox.header.frame_id = "livox_frame"
            point_livox.header.stamp = self.get_clock().now().to_msg()
            point_livox.point.x = obj.object.kinematics.pose_with_covariance.pose.position.x
            point_livox.point.y = obj.object.kinematics.pose_with_covariance.pose.position.y
            point_livox.point.z = obj.object.kinematics.pose_with_covariance.pose.position.z

            try:
                # TF transform (livox_frame -> map)
                point_map = self.tf_buffer.transform(point_livox, "map", timeout=rclpy.duration.Duration(seconds=0.5))
                x_centers.append(point_map.point.x)
                y_centers.append(point_map.point.y)
            except Exception as e:
                self.get_logger().warn(f"TF transform failed: {e}")
                continue

        # --- Frenet 변환 ---
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
