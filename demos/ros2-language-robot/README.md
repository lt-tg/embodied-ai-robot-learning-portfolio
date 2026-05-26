# Demo 3: ROS2 Language-Controlled Robot System

Goal: show robotics system integration: language command parsing, ROS2-style interfaces, policy bridge, and replay/debugging.

## MVP

- Parse natural-language commands into structured robot tasks.
- Publish the task to a policy bridge node.
- Return a structured result.
- Record a rosbag when running in a real ROS2 environment.

## Directory Contract

```text
demos/ros2-language-robot/
  src/task_parser/
  src/policy_bridge/
  launch/
  bags/
  configs/
```

## Interface Sketch

Topic:
- `/language_command`: raw command string
- `/structured_task`: `{action, object, target}`

Action:
- `/execute_task`: sends structured task, receives status and error message

Launch:
- `language_robot.launch.py`: parser + policy bridge + simulator bridge

## Local Starter

Before installing ROS2, use the tested parser in the root package:

```powershell
cd robot-learning-lab
python examples/day25_parse_command.py
```

## Resume Bullet

Developed a ROS2 language-to-action interface that parses natural-language commands into structured robot tasks and executes them through simulated robot policy nodes with rosbag-based replay and debugging.
