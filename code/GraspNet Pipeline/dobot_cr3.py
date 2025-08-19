import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'TCP-IP-Python-V3'))

from dobot_api import DobotApiDashboard, DobotApiMove, DobotApi, DobotApiFeedBack
import time

class DobotController:
    def __init__(self, ip="255.255.255.255"):
        self.ip = ip
        self.dashboard_port = 29999
        self.move_port = 30003
        self.feed_port = 30004
        self.user_index = 0
        self.tool_index = 2

        self.dashboard = None
        self.move = None
        self.feed = None
        self.feed_back = None

    def connect(self):
        print("Connecting DoBot...")
        try:
            self.dashboard = DobotApiDashboard(self.ip, self.dashboard_port)
            self.move = DobotApiMove(self.ip, self.move_port)
            self.feed = DobotApi(self.ip, self.feed_port)
            self.feed_back = DobotApiFeedBack(self.ip, self.feed_port)
            self.dashboard.Tool(self.tool_index)
            print("Successfully connected to DoBot.")
            return True
        except Exception as e:
            print(f"Failed: {e}")
            return False

    def clear_error_and_enable(self):
        self.dashboard.ClearError()
        self.dashboard.EnableRobot()
        time.sleep(1)
        self.dashboard.SpeedFactor(50)

    def get_pose(self):
        result = self.dashboard.GetPose()
        parts = result.split("{")[-1].split("}")[0]
        values = [float(x.strip()) for x in parts.split(",")]
        return values

    def move_to(self, x, y, z, rx, ry, rz):
        self.move.MovJ(x, y, z, rx, ry, rz)

    def move_joint(self, joint1, joint2, joint3, joint4, joint5, joint6):
        self.move.JointMovJ(joint1, joint2, joint3, joint4, joint5, joint6)

    def sync(self):
        self.move.Sync()

    def init_gripper(self):
        self.dashboard.ModbusCreate("127.0.0.1", 60000, 1, 1)
        self.dashboard.SetHoldRegs(0, 256, 1, "165", "U16")
        self.dashboard.SetHoldRegs(0, 260, 1, "90", "U16")
        self.dashboard.SetHoldRegs(0, 257, 1, "35", "U16")
        time.sleep(3)

    def open_gripper(self, strength=10):
        value = str(strength * 10)
        self.dashboard.SetHoldRegs(0, 259, 1, value, "U16")

    def set_tool_coordinates(self, tool_number=2):
        self.dashboard.Tool(tool_number)

    def shutdown(self):
        self.dashboard.DisableRobot()
        time.sleep(1)
        print("Dobot closed")

