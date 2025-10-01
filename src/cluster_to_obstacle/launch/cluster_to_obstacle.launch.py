import os
import launch
import launch_ros.actions

def generate_launch_description():
    # Launch arguments
    frenet_waypoints_topic = launch.substitutions.LaunchConfiguration(
        'frenet_waypoints_topic', default='/global_waypoints')
    cluster_topic = launch.substitutions.LaunchConfiguration(
        'cluster_topic', default='/clusters')
    obstacle_pub_topic = launch.substitutions.LaunchConfiguration(
        'obstacle_pub_topic', default='/perception/detection/raw_obstacles')

    # Cluster to Obstacle Node
    cluster_to_obstacle_node = launch_ros.actions.Node(
        package='cluster_to_obstacle',
        executable='cluster_to_obstacle',
        name='cluster_to_obstacle',
        output='screen',
        parameters=[{
            'frenet_waypoints_topic': frenet_waypoints_topic,
            'cluster_topic': cluster_topic,
            'obstacle_pub_topic': obstacle_pub_topic
        }]
    )

    tf = launch_ros.actions.Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            "0.3", "0.0", "0.0", "0.0", "0.0", "0.7071", "0.7071", "base_link", "livox_frame"]
    )

    return launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            'frenet_waypoints_topic', default_value='/global_waypoints'),
        launch.actions.DeclareLaunchArgument(
            'cluster_topic', default_value='/clusters'),
        launch.actions.DeclareLaunchArgument(
            'obstacle_pub_topic', default_value='/perception/detection/raw_obstacles'),
        cluster_to_obstacle_node,
        tf
    ])
