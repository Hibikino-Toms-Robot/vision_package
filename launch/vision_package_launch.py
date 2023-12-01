from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
import os

def generate_launch_description():
    img_config = os.path.join(
      get_package_share_directory('vision_package'),
      'config',
      'vision_package.yaml'
    )
    yolov5_config = os.path.join(
      get_package_share_directory('vision_package'),
      'config',
      'yolov5.yaml'
    )

    return LaunchDescription([
        Node(
            package='vision_package',
            executable='vision_service',
            name='vision_service',
            parameters = [yolov5_config],
            output = 'screen',
        ),

        # Node(
        #         package="rviz2",
        #         executable="rviz2",
        #         name="rviz2",
        #         arguments=["-d", rviz_config_dir],
        #         output="screen",
        # ),
        # ExecuteProcess(
        #         cmd=[
        #             "ros2",
        #             "bag",
        #             "record",
        #             "-a",
        #             "-o",
        #             rosbag_directory,
        #             "--compression-mode",
        #             "file",
        #             "--compression-format",
        #             "zstd",
        #         ],
        #         output="screen",
        #         condition=IfCondition(record),
        #         shell=True
        # ),
    ])
