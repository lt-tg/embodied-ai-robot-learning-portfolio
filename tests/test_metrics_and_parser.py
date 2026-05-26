import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_learning_lab.metrics import action_mse, success_rate
from robot_learning_lab.task_parser import parse_robot_command


class MetricsAndParserTests(unittest.TestCase):
    def test_success_rate_counts_truthy_success_flags(self):
        episodes = [{"success": True}, {"success": False}, {"success": 1}, {"success": 0}]

        self.assertAlmostEqual(success_rate(episodes), 0.5)

    def test_action_mse_averages_over_all_action_dimensions(self):
        predicted = [[1.0, 2.0], [3.0, 4.0]]
        target = [[1.0, 1.0], [1.0, 2.0]]

        self.assertAlmostEqual(action_mse(predicted, target), 2.25)

    def test_command_parser_extracts_pick_and_place_slots(self):
        command = parse_robot_command("pick the red cube and place it on the blue plate")

        self.assertEqual(command.action, "pick_place")
        self.assertEqual(command.object, "red cube")
        self.assertEqual(command.target, "blue plate")

    def test_command_parser_supports_chinese_pick_and_place(self):
        command = parse_robot_command("把红色方块放到蓝色盘子上")

        self.assertEqual(command.action, "pick_place")
        self.assertEqual(command.object, "红色方块")
        self.assertEqual(command.target, "蓝色盘子")


if __name__ == "__main__":
    unittest.main()
