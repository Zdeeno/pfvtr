tmux kill-session -t bearnav

tmux new-session -d -s "bearnav" -n "bearnav"
tmux new-window -d -n "mapmaker"
tmux new-window -d -n "repeater"
tmux new-window -d -n "misc"
tmux new-window -d -n "maps"

tmux send-keys -t bearnav:bearnav "source ../../devel/setup.bash" Enter
tmux send-keys -t bearnav:bearnav "roslaunch bearnav2 bearnav2.launch "
tmux send-keys -t bearnav:mapmaker "source ../../devel/setup.bash" Enter
tmux send-keys -t bearnav:mapmaker "rostopic pub /bearnav2/mapmaker/goal "
tmux send-keys -t bearnav:repeater "source ../../devel/setup.bash" Enter
tmux send-keys -t bearnav:repeater "rostopic pub /bearnav2/repeater/goal "
tmux send-keys -t bearnav:misc "source ../../devel/setup.bash" Enter
tmux send-keys -t bearnav:misc "timeout 3 rostopic hz /camera_front/image_color" Enter
sleep 3
tmux send-keys -t bearnav:misc "timeout 3 rostopic hz /husky_velocity_controller/odom" Enter
tmux send-keys -t bearnav:maps "cd ~/.ros" Enter
