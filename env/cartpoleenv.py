import numpy as np
import gymnasium as gym
from gymnasium.utils import seeding
from gymnasium import spaces, logger
import time

import sys
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from . import cartpole_sim

class CartPoleEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, action_type='discrete'):
        super(CartPoleEnv, self).__init__()
        self.cart_pole_sim_model = cartpole_sim.CartPoleSimModel("/prismatic_joint", "/revolute_joint")
        self.action_type = action_type
        self.push_force = 0
        self.q = [0.0, 0.0]
        self.q_last = [0.0, 0.0]

        self.theta_max = 40*np.pi / 360
        self.cart_pos_max = 0.8

        high = np.array(
            [
                self.cart_pos_max,
                self.theta_max,
                1000000.0,
                1000000.0
            ],
            dtype=np.float32,
        )


        if self.action_type == 'discrete':
            self.action_space = spaces.Discrete(3)
        elif self.action_type == 'continuous':
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            assert 0, "The action type \'" + self.action_type + "\' can not be recognized"

        print(f"Action Space:{self.action_type}")
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.counts = 0
        self.steps_beyond_done = None

        self.cart_pole_sim_model.sim.startSimulation()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if self.action_type == 'discrete':
            assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        q = [0.0, 0.0]
        q[0] , q[1]= self.cart_pole_sim_model.getJointPosition()

        v = [0.0, 0.0]
        v[0], v[1]= self.cart_pole_sim_model.getJointVelocity()
        self.q_last = self.q
        self.q = q

        if self.action_type == 'discrete':
            if action == 0:
                self.push_force = 0
            elif action == 1:
                self.push_force = 1.0
            elif action == 2:
                self.push_force = -1.0
        elif self.action_type == 'continuous':
            self.push_force = action[0]*2.0 # The action is in [-1.0, 1.0], therefore the force is in [-2.0, 2.0]
        else:
            assert 0, "The action type \'" + self.action_type + "\' can not be recognized"

        # set action
        self.cart_pole_sim_model.setJointForce(self.push_force)
       
        done = (q[0] <= -self.cart_pos_max) or (q[0] >= self.cart_pos_max) or (q[1] < -self.theta_max) or (q[1] > self.theta_max)
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0


        self.state = (self.q[0], self.q[1], v[0], v[1])
        self.counts += 1

        self.cart_pole_sim_model.sim.step()
        return np.array(self.state, dtype=np.float32), reward, done, False, {}
    
    def reset(self, seed=None):
        # print('Reset the environment after {} counts'.format(self.counts))
        self.counts = 0
        self.push_force = 0
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None

        self.cart_pole_sim_model.sim.stopSimulation() # stop the simulation
        time.sleep(0.1) # ensure the coppeliasim is stopped
        self.cart_pole_sim_model.sim.setStepping(True)

        self.cart_pole_sim_model.setJointPosition([self.state[0], self.state[1]])
        self.cart_pole_sim_model.sim.startSimulation()

        self.cart_pole_sim_model.setJointForce(0)

        return np.array(self.state, dtype=np.float32), {}
    
    def render(self):
        return None

    def close(self):
        self.cart_pole_sim_model.sim.stopSimulation() # stop the simulation
        print('Close the environment')
        return None

if __name__ == "__main__":
    env = CartPoleEnv()
    env.reset()

    for _ in range(500):
        action = env.action_space.sample() # random action
        env.step(action)
        print(env.state)

    env.close()