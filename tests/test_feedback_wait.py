import unittest
from unittest.mock import Mock, patch

from robot.feedback_wait import wait_for_target


class FeedbackWaitTests(unittest.TestCase):
    @patch('robot.feedback_wait.time.sleep')
    def test_delayed_and_transient_feedback(self, sleep):
        read = Mock(side_effect=[[0], [.07], [0], [.07], [.069], [.07]])
        self.assertTrue(wait_for_target(read, [.07], .002))
        self.assertEqual(read.call_count, 6)

    def test_invalid_or_missing_feedback_is_not_success(self):
        for value in (None, [], [float('nan')], [0]):
            self.assertFalse(wait_for_target(lambda: value, [.07], .002, timeout=0))

    def test_disconnect_propagates(self):
        with self.assertRaises(ConnectionError):
            wait_for_target(Mock(side_effect=ConnectionError()), [.07], .002)


class GripperCompletionTests(unittest.TestCase):
    def setUp(self):
        from robot.play_env import PlayRobotEnv
        self.env = PlayRobotEnv.__new__(PlayRobotEnv)
        self.env.robot = Mock()
        self.env.config = {'grasp': {'gripper_max_width': .07}}
        self.env.gripper_open_width = .07

    @patch('robot.play_env.wait_for_target', return_value=True)
    def test_open_clamps_to_hardware_limit_and_verifies(self, wait):
        self.assertTrue(self.env.open_gripper(.08))
        self.env.robot.set_gripper.assert_called_once_with(position=.07)
        wait.assert_called_once_with(self.env.robot.left.get_eef_pos, [.07], .002)

    @patch('robot.play_env.wait_for_target', return_value=False)
    def test_acceptance_without_completion_is_failure(self, wait):
        self.assertFalse(self.env.open_gripper())

    @patch('robot.play_env.wait_for_target')
    def test_rejected_command_does_not_wait(self, wait):
        self.env.robot.set_gripper.return_value = False
        self.assertFalse(self.env.open_gripper())
        wait.assert_not_called()


if __name__ == '__main__':
    unittest.main()
