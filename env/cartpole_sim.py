import sys
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

class CartPoleSimModel():
    def __init__(self, prismatic_joint_name, revolute_joint_name) -> None:
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject("sim")
        self.sim.setStepping(True)
        print("Initializing Environment........")

        self.prismatic_joint_hd = self.sim.getObject(prismatic_joint_name)
        print(f"Get Joint {prismatic_joint_name}")
        self.revolute_joint_hd = self.sim.getObject(revolute_joint_name)
        print(f"Get Joint {revolute_joint_name}")

    def getJointPosition(self):
        pris_pos = self.sim.getJointPosition(self.prismatic_joint_hd)
        rev_pos = self.sim.getJointPosition(self.revolute_joint_hd)
        return pris_pos, rev_pos

    def getJointVelocity(self):
        pris_vel = self.sim.getJointVelocity(self.prismatic_joint_hd)
        rev_vel = self.sim.getJointPosition(self.revolute_joint_hd)
        return pris_vel, rev_vel

    def setJointPosition(self, pos):
        self.sim.setJointPosition(self.prismatic_joint_hd, pos[0])
        self.sim.setJointPosition(self.revolute_joint_hd, pos[1])

    def setJointVelocity(self, vel):
        self.sim.setJointTargetVelocity(self.prismatic_joint_hd, vel[0])
        self.sim.setJointTargetVelocity(self.revolute_joint_hd, vel[1])

    def setJointForce(self,force):
        self.sim.setJointTargetForce(self.prismatic_joint_hd, force)
        
    
