from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
import launch_ros.actions
import os
import yaml
from launch.substitutions import EnvironmentVariable
import pathlib
import launch.actions
from launch.actions import DeclareLaunchArgument
from pathlib import Path


def generate_launch_description():
    base_path = Path(get_package_share_directory("roar_carla_manual_controller"))
    config_path: Path = base_path / "params" / "config.yaml"
    assert config_path.exists(), f"{config_path} does not exist"
    return LaunchDescription(
        [
            launch_ros.actions.Node(
                package="roar_carla_manual_controller",
                executable="roar_carla_manual_controller_node",
                name="roar_carla_manual_controller",
                parameters=[config_path.as_posix()],
                remappings=[
                    ("roar_carla_cmd", "/sim/vehicle/control"),
                    ("image", "/roar/front/rgb/image"),
                    ("l_image", "/roar/front/rgb/l_image"),
                    ("r_image", "/roar/front/rgb/r_image"),
                ],
            ),
        ]
    )
