import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class HopperEnvRandDir(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self._goal_vel = self.sample_goals()  # *modification*
        self._goal_direction = -1.0 if self._goal_vel < 1.5 else 1.0  # *modification*
        mujoco_env.MujocoEnv.__init__(self, 'hopper.xml', 4)
        utils.EzPickle.__init__(self)


    def sample_goals(self):
        # *modification*
        return np.random.uniform(0.0, 3.0)

    def step(self, a):
        # print(a)
        posbefore = self.sim.data.qpos[0]
        # print("before" + str(posbefore))
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        # print("after" + str(posafter))
        alive_bonus = 1.0
        reward = self._goal_direction * (posafter - posbefore) / self.dt
        # print("reward "  + str(reward))
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()


        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset(self):
        # *modification*
        self._goal_vel = self.sample_goals()
        self._goal_direction = -1.0 if self._goal_vel < 1.5 else 1.0
        self.sim.reset()
        ob = self.reset_model()
        return ob

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20