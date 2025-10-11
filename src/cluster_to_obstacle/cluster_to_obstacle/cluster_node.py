

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, HistoryPolicy
from tier4_perception_msgs.msg import DetectedObjectsWithFeature
from f110_msgs.msg import ObstacleArray, Obstacle as ObstacleMessage
from f110_msgs.msg import WpntArray
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
import numpy as np
from tf_transformations import euler_from_quaternion, quaternion_from_euler
# TF2 imports
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PointStamped
import tf2_geometry_msgs.tf2_geometry_msgs  # PointStamped 변환용
# Frenet 변환 클래스
from frenet_conversion.frenet_converter import FrenetConverter
from visualization_msgs.msg import Marker, MarkerArray
from bisect import bisect_left
import math

latching_qos = QoSProfile(
    depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST)

class ClusterToObstacle(Node):
    def __init__(self):
        super().__init__('cluster_to_obstacle')

        # --- Parameters ---
        self.declare_parameter("frenet_waypoints_topic", "/global_waypoints")
        self.declare_parameter("cluster_topic", "/clusters")
        self.declare_parameter("obstacle_pub_topic", "/perception/detection/raw_obstacles")
        self.declare_parameter("obstacle_marker_topic", "/perception/detection/obstacles_markers")

        self.declare_parameter("min_obs_size", 0.05)            
        self.declare_parameter("max_obs_size", 0.5)             
        self.declare_parameter("max_viewing_distance", 5.0)     
        self.declare_parameter("boundaries_inflation", 0.1)     

        self.min_obs_size = self.get_parameter("min_obs_size").value
        self.max_obs_size = self.get_parameter("max_obs_size").value
        self.max_viewing_distance = self.get_parameter("max_viewing_distance").value
        self.boundaries_inflation = self.get_parameter("boundaries_inflation").value

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
            "/global_waypoints",
            self.pathCb,
            10
        )

        self.car_s = 0.0
        self.car_state_sub = self.create_subscription(
            Odometry,
            '/car_state/frenet/odom',
            self.car_state_callback,
            10
        )

        # --- Publishers ---
        self.obstacles_pub = self.create_publisher(
            ObstacleArray,
            self.get_parameter("obstacle_pub_topic").value,
            10
        )
        self.boundaries_pub = self.create_publisher(
        Marker, '/perception/detect_bound', qos_profile=latching_qos)

        self.obstacle_marker_pub = self.create_publisher(
            MarkerArray,
            self.get_parameter("obstacle_marker_topic").value,
            10
        )

        # --- TF2 ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- Track boundary arrays (initialized after receiving waypoints) ---
        self.s_array = None
        self.d_right_array = None
        self.d_left_array = None
        self.track_length = None
        self.smallest_d = None
        self.biggest_d = None

        self.get_logger().info("[cluster_to_obstacle] node initialized")

    def initialize_converter(self):
        converter = FrenetConverter(
            self.waypoints[:, 0], self.waypoints[:, 1], self.waypoints[:, 2])
        self.get_logger().info(
            "[Opponent Detection]: initialized FrenetConverter object")
        return converter
        
    # --- Utilities (from Detect.py logic) ---
    def normalize_s(self, x: float, track_length: float) -> float:
        x = x % (track_length)
        if x > track_length/2:
            x -= track_length
        return x

    def laserPointOnTrack(self, s: float, d: float, car_s: float) -> bool:
        # Requires self.track_length, self.s_array, self.d_right_array, self.d_left_array, self.smallest_d, self.biggest_d

        if self.normalize_s(s - car_s, self.track_length) > self.max_viewing_distance:
            return False
        if abs(d) >= self.biggest_d:
            return False
        if abs(d) <= self.smallest_d:
            return True
        idx = bisect_left(self.s_array, s)
        if idx:
            idx -= 1
        # boundary arrays are stored as right/left (with boundary inflation already subtracted)
        if d <= -self.d_right_array[idx] or d >= self.d_left_array[idx]:
            return False
        return True

    # --- Callbacks ---
    def pathCb(self, data):
        # Initial calls: initialize the converter
        if self.converter is None:
            self.waypoints = np.array(
                [[wpnt.x_m, wpnt.y_m, wpnt.psi_rad] for wpnt in data.wpnts])
            self.converter = self.initialize_converter()

        # Second call: create the boundaries arrays
        if self.s_array is None or self.path_needs_update:
            self.get_logger().info(
                '[Opponent Detection]: received global path')
            waypoint_array = data.wpnts
            points = []
            self.s_array = []
            self.d_right_array = []
            self.d_left_array = []
            for waypoint in waypoint_array:
                self.s_array.append(waypoint.s_m)
                self.d_right_array.append(
                    waypoint.d_right-self.boundaries_inflation)
                self.d_left_array.append(
                    waypoint.d_left-self.boundaries_inflation)
                resp = self.converter.get_cartesian(
                    waypoint.s_m, -waypoint.d_right+self.boundaries_inflation)
                points.append(Point(x=resp[0], y=resp[1], z=0.0))
                resp = self.converter.get_cartesian(
                    waypoint.s_m, waypoint.d_left-self.boundaries_inflation)
                points.append(Point(x=resp[0], y=resp[1], z=0.0))
            self.smallest_d = min(self.d_right_array+self.d_left_array)
            self.biggest_d = max(self.d_right_array+self.d_left_array)
            self.track_length = data.wpnts[-1].s_m

            self.detection_boundaries_marker = Marker()
            self.detection_boundaries_marker.header.frame_id = "map"
            self.detection_boundaries_marker.header.stamp = self.get_clock().now().to_msg()
            self.detection_boundaries_marker.id = 0
            self.detection_boundaries_marker.type = Marker.SPHERE_LIST
            self.detection_boundaries_marker.scale.x = 0.02
            self.detection_boundaries_marker.scale.y = 0.02
            self.detection_boundaries_marker.scale.z = 0.02
            self.detection_boundaries_marker.color.a = 1.
            self.detection_boundaries_marker.color.g = 0.
            self.detection_boundaries_marker.color.r = 1.
            self.detection_boundaries_marker.color.b = 0.
            self.detection_boundaries_marker.points = points

        # Republish markers
        self.boundaries_pub.publish(self.detection_boundaries_marker)
        self.path_needs_update = False

    def car_state_callback(self, msg: Odometry):
        # In Detect.py car_s = data.pose.pose.position.x
        try:
            self.car_s = msg.pose.pose.position.x
        except Exception:
            self.get_logger().warn("car_state message does not contain expected pose.pose.position.x")

    def cluster_callback(self, msg: DetectedObjectsWithFeature):
        if self.converter is None:
            self.get_logger().warn("FrenetConverter not initialized yet; dropping clusters")
            return
        if self.s_array is None:
            self.get_logger().warn("Track boundaries not initialized yet; dropping clusters")
            return

        obs_array_msg = ObstacleArray()
        obs_array_msg.header.stamp = self.get_clock().now().to_msg()
        obs_array_msg.header.frame_id = "map"

        marker_array = MarkerArray()

        valid_objs = []
        for obj in msg.feature_objects:
            point_livox = PointStamped()
            point_livox.header.frame_id = "livox_frame"
            point_livox.header.stamp = rclpy.time.Time().to_msg()
            point_livox.point.x = obj.object.kinematics.pose_with_covariance.pose.position.x
            point_livox.point.y = obj.object.kinematics.pose_with_covariance.pose.position.y
            point_livox.point.z = obj.object.kinematics.pose_with_covariance.pose.position.z

            try:
                point_map = self.tf_buffer.transform(point_livox, "map")
            except Exception as e:
                self.get_logger().debug(f"TF transform failed for an object: {e}")
                continue

            x_map = point_map.point.x
            y_map = point_map.point.y
            try:
                size = max(obj.object.shape.dimensions.x, obj.object.shape.dimensions.y)
            except Exception:
                size = 0.0
                self.get_logger().warn("Detected object missing shape.dimensions; size set to 0.0")

            valid_objs.append({'obj': obj, 'x': x_map, 'y': y_map, 'size': size})

        if len(valid_objs) > 0:
            xs = np.array([v['x'] for v in valid_objs])
            ys = np.array([v['y'] for v in valid_objs])
            s_arr, d_arr = self.converter.get_frenet(xs, ys)

            published_count = 0
            for idx, v in enumerate(valid_objs):
                s = s_arr[idx]
                d = d_arr[idx]
                size = v['size']

                if not self.laserPointOnTrack(s, d, self.car_s):
                    continue
                # if size < self.min_obs_size or size > self.max_obs_size:
                #     continue

                obs_msg = ObstacleMessage()
                obs_msg.id = published_count
                obs_msg.s_center = float(s)
                obs_msg.d_center = float(d)
                obs_msg.size = float(size)
                obs_msg.s_start = float(s - size/2.0)
                obs_msg.s_end = float(s + size/2.0)
                obs_msg.d_left = float(d + size/2.0)
                obs_msg.d_right = float(d - size/2.0)

                obs_array_msg.obstacles.append(obs_msg)

                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.id = published_count
                marker.type = Marker.CUBE
                marker.scale.x = size
                marker.scale.y = size
                marker.scale.z = size
                marker.color.a = 0.5
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.pose.position.x = v['x']
                marker.pose.position.y = v['y']
                marker.pose.position.z = 0.0
                q = quaternion_from_euler(0, 0, 0)
                marker.pose.orientation.x = q[0]
                marker.pose.orientation.y = q[1]
                marker.pose.orientation.z = q[2]
                marker.pose.orientation.w = q[3]
                marker_array.markers.append(marker)

                published_count += 1

            self.obstacle_marker_pub.publish(marker_array)
            self.get_logger().info(f"Published {published_count} obstacles with track filtering")

        # ✅ 무조건 ObstacleArray는 퍼블리시
        self.obstacles_pub.publish(obs_array_msg)



def main():
    rclpy.init()
    node = ClusterToObstacle()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt")
        
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
