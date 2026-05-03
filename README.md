##### comandi utili #####


cd ~/ros_ws

rm -rf build/ install/ log/

colcon build --symlink-install

source install/setup.bash

ros2 launch storm_description sim.launch.py


