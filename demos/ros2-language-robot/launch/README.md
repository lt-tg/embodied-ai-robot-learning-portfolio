# Launch Plan

Add `language_robot.launch.py` after ROS2 is installed.

Expected nodes:

- `task_parser_node`: subscribes to `/language_command`, publishes `/structured_task`.
- `policy_bridge_node`: receives structured task and calls the policy model.
- `sim_bridge_node`: forwards actions to MuJoCo/Gazebo/Isaac.

Keep the first implementation minimal: one command, one task, one simulator.
