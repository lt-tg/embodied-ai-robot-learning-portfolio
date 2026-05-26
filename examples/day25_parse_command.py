import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_learning_lab.task_parser import parse_robot_command


if __name__ == "__main__":
    commands = [
        "pick the red cube and place it on the blue plate",
        "把红色方块放到蓝色盘子上",
        "pick green block",
    ]

    for text in commands:
        parsed = parse_robot_command(text)
        print(f"{text} -> action={parsed.action}, object={parsed.object}, target={parsed.target}")
