tmux kill-session -t bearnav

tmux new-session -d -s "bearnav" -n "bearnav"
tmux new-window -d -n "mapmaker"
tmux new-window -d -n "repeater"
tmux new-window -d -n "repr"
tmux new-window -d -n "misc"
tmux new-window -d -n "maps"
tmux new-window -d -n "viz"

x=$(echo $SHELL | sed 's:.*/::')

tmux send-keys -t bearnav:bearnav "source ../../devel/setup.$x" Enter
tmux send-keys -t bearnav:bearnav "roslaunch pfvtr sim.launch "
tmux send-keys -t bearnav:repr "source ../../devel/setup.$x" Enter
tmux send-keys -t bearnav:repr "roslaunch pfvtr repr-spot2.launch "
tmux send-keys -t bearnav:mapmaker "source ../../devel/setup.$x" Enter
tmux send-keys -t bearnav:mapmaker "rostopic pub /pfvtr/mapmaker/goal "
tmux send-keys -t bearnav:repeater "source ../../devel/setup.$x" Enter
tmux send-keys -t bearnav:repeater "rostopic pub /pfvtr/repeater/goal "
tmux send-keys -t bearnav:misc "source ../../devel/setup.$x" Enter
tmux send-keys -t bearnav:misc "timeout 3 rostopic hz /camera_front/image_color" Enter
sleep 3
tmux send-keys -t bearnav:misc "timeout 3 rostopic hz /husky_velocity_controller/odom" Enter
tmux send-keys -t bearnav:maps "cd ~/.ros" Enter
tmux send-keys -t bearnav:viz "source ../../devel/setup.$x" Enter
tmux send-keys -t bearnav:viz "python ./src/gui/particles-viz.py"
