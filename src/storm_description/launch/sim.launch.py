import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    # Recupera il percorso del pacchetto
    pkg_path = get_package_share_directory('storm_description')
    
    # 1. Processa il file XACRO per la descrizione del robot STORM
    xacro_file = os.path.join(pkg_path, 'urdf', 'storm_robot.urdf.xacro')
    robot_description_config = xacro.process_file(xacro_file).toxml()

    # 2. Percorso della Mappa A (Map 1: corridoio da 1.2m) 
    #world_path = os.path.join(pkg_path, 'worlds', 'map_a.world')#
    
    world_path= os.path.join(pkg_path,'worlds','training_map.world')

    # 3. Azione per avviare Gazebo caricando direttamente la mappa 
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')]),
        launch_arguments={'gz_args': f'-r {world_path}'}.items(),
    )

    # 4. Nodo per inserire (spawnare) il robot STORM nella mappa 
    # Posizionato al centro del corridoio (x: 0, y: 0) 
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description', 
            '-name', 'storm', 
            '-x', '-2.3',    # Centro esatto del nuovo corridoio d'ingresso
            '-y', '-3.0',     # Posizione all'inizio del corridoio (in basso)
            '-z', '0.5',      # Spawna a 50cm di altezza per cadere dolcemente senza incastrarsi
            '-Y', '1.57'      # Ruotato per guardare dritto in avanti lungo il corridoio (+Y)
        ],
        output='screen'
    )

    # 5. Pubblica lo stato del robot per le trasformazioni (TF)
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_config, 
            'use_sim_time': True,
        }],
        remappings=[
           ('/joint_states', '/model/storm/joint_states'),
        ],
    )

    # 6. Bridge ROS-GZ per sensori e attuatori 
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{'use_sim_time': True}],
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
            '/model/storm/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist',
            '/model/storm/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan',
            '/model/storm/joint_states@sensor_msgs/msg/JointState[ignition.msgs.Model',
            '/model/storm/joint/vtm_joint/0/cmd_pos@std_msgs/msg/Float64]ignition.msgs.Float',
            '/model/storm/tf@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V',
            '/world/map_a/control@ros_gz_interfaces/srv/ControlWorld',
            '/world/map_a/set_pose@ros_gz_interfaces/srv/SetEntityPose',
        ],

        output='screen'
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    return LaunchDescription([
        gazebo,
        node_robot_state_publisher,
        spawn_robot,
        bridge,
        rviz_node
    ])