import zerorpc
from polymetis import RobotInterface
import scipy.spatial.transform as st
import numpy as np
import torch

class FrankaInterface:
    def __init__(self):
        self.robot = RobotInterface(ip_address='localhost', enforce_version=False)

    def get_ee_pose(self):
        data = self.robot.get_ee_pose()
        pos = data[0].numpy()
        quat_xyzw = data[1].numpy()
        rot_vec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
        return np.concatenate([pos, rot_vec]).tolist()
    
    def get_joint_positions(self):
        return self.robot.get_joint_positions().numpy().tolist()
    
    def get_joint_velocities(self):
        return self.robot.get_joint_velocities().numpy().tolist()
    
    def move_to_joint_positions(self, positions, time_to_go):
        positions = np.asarray(positions, dtype=np.float32)
        self.robot.move_to_joint_positions(
            positions=torch.tensor(positions, dtype=torch.float32),
            time_to_go=float(time_to_go)
        )
    
    def start_cartesian_impedance(self, Kx, Kxd):
        self.robot.start_cartesian_impedance(
            Kx=torch.tensor(Kx, dtype=torch.float32),
            Kxd=torch.tensor(Kxd, dtype=torch.float32)
        )

    def update_desired_ee_pose(self, pose):
        pose = np.asarray(pose, dtype=np.float32)
        self.robot.update_desired_ee_pose(
            position=torch.tensor(pose[:3], dtype=torch.float32),
            orientation=torch.tensor(st.Rotation.from_rotvec(pose[3:]).as_quat(), dtype=torch.float32)
        )

    def go_home(self):
        self.robot.go_home()

    def terminate_current_policy(self):
        self.robot.terminate_current_policy()

s = zerorpc.Server(FrankaInterface())
s.bind("tcp://0.0.0.0:4242")
s.run()