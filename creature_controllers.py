import pydart2 as pydart
import numpy as np
import math

# vector: [dPhase, amplitude, period]
class Controller(object):
    def __init__(self, skel, vector):
        self.skel = skel
        self.target = None
        self.Kp = np.array([0.0] * 6 + [400.0] * (self.skel.ndofs - 6))
        self.Kd = np.array([0.0] * 6 + [40.0] * (self.skel.ndofs - 6))
        self.dPhase = vector[0]
        self.amp = vector[1]
        self.period = vector[2]
        #add bounds on angles
        self.minJointAngle = -math.pi/2
        self.maxJointAngle = math.pi/2

    def update_target_poses(self):
        skel = self.skel
        pose = self.skel.q
        pose[('joint_2', 'joint_3', 'joint_4', 'joint_5')] = self.periodic(self.amp, self.period, 0), self.periodic(self.amp, self.period, 1*self.dPhase),\
                                                             self.periodic(self.amp, self.period, 2*self.dPhase), self.periodic(self.amp, self.period, 3*self.dPhase)
        return pose

    def periodic(self, amplitude, period, phase):
        return (amplitude * np.sin(period * self.skel.world.t + phase))


    def compute(self):
        self.target = self.update_target_poses()
        return -self.Kp * (self.skel.q - self.target) - self.Kd * self.skel.dq