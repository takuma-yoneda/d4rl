""" A pointmass maze env."""
from gym.envs.mujoco import mujoco_env
from gym import utils
from d4rl import offline_env
from d4rl.pointmaze.dynamic_mjc import MJCModel
import numpy as np
import random


WALL = 10
EMPTY = 11
GOAL = 12


def parse_maze(maze_str):
    lines = maze_str.strip().split('\\')
    width, height = len(lines), len(lines[0])
    maze_arr = np.zeros((width, height), dtype=np.int32)
    for w in range(width):
        for h in range(height):
            tile = lines[w][h]
            if tile == '#':
                maze_arr[w][h] = WALL
            elif tile == 'G':
                maze_arr[w][h] = GOAL
            elif tile == ' ' or tile == 'O' or tile == '0':
                maze_arr[w][h] = EMPTY
            else:
                raise ValueError('Unknown tile type: %s' % tile)
    return maze_arr


def point_maze(maze_str):
    maze_arr = parse_maze(maze_str)

    mjcmodel = MJCModel('point_maze')
    mjcmodel.root.compiler(inertiafromgeom="true", angle="radian", coordinate="local")
    mjcmodel.root.option(timestep="0.01", gravity="0 0 0", iterations="20", integrator="Euler")
    default = mjcmodel.root.default()
    default.joint(damping=1, limited='false')
    default.geom(friction=".5 .1 .1", density="1000", margin="0.002", condim="1", contype="2", conaffinity="1")

    asset = mjcmodel.root.asset()
    asset.texture(type="2d",name="groundplane",builtin="checker",rgb1="0.2 0.3 0.4",rgb2="0.1 0.2 0.3",width=100,height=100)
    asset.texture(name="skybox",type="skybox",builtin="gradient",rgb1=".4 .6 .8",rgb2="0 0 0",
               width="800",height="800",mark="random",markrgb="1 1 1")
    asset.material(name="groundplane",texture="groundplane",texrepeat="20 20")
    asset.material(name="wall",rgba=".7 .5 .3 1")
    asset.material(name="target",rgba=".6 .3 .3 1")

    visual = mjcmodel.root.visual()
    visual.headlight(ambient=".4 .4 .4",diffuse=".8 .8 .8",specular="0.1 0.1 0.1")
    visual.map(znear=.01)
    visual.quality(shadowsize=2048)

    worldbody = mjcmodel.root.worldbody()
    worldbody.geom(name='ground',size="40 40 0.25",pos="0 0 -0.1",type="plane",contype=1,conaffinity=0,material="groundplane")

    particle = worldbody.body(name='particle', pos=[1.2,1.2,0])
    particle.geom(name='particle_geom', type='sphere', size=0.1, rgba='0.0 0.0 1.0 0.0', contype=1)
    particle.site(name='particle_site', pos=[0.0,0.0,0], size=0.2, rgba='0.3 0.6 0.3 1')
    particle.joint(name='ball_x', type='slide', pos=[0,0,0], axis=[1,0,0])
    particle.joint(name='ball_y', type='slide', pos=[0,0,0], axis=[0,1,0])

    worldbody.site(name='target_site', pos=[0.0,0.0,0], size=0.2, material='target')

    width, height = maze_arr.shape
    for w in range(width):
        for h in range(height):
            if maze_arr[w,h] == WALL:
                worldbody.geom(conaffinity=1,
                               type='box',
                               name='wall_%d_%d'%(w,h),
                               material='wall',
                               pos=[w+1.0,h+1.0,0],
                               size=[0.5,0.5,0.2])

    actuator = mjcmodel.root.actuator()
    actuator.motor(joint="ball_x", ctrlrange=[-1.0, 1.0], ctrllimited=True, gear=100)
    actuator.motor(joint="ball_y", ctrlrange=[-1.0, 1.0], ctrllimited=True, gear=100)

    return mjcmodel


LARGE_MAZE = \
        "############\\"+\
        "#OOOO#OOOOO#\\"+\
        "#O##O#O#O#O#\\"+\
        "#OOOOOO#OOO#\\"+\
        "#O####O###O#\\"+\
        "#OO#O#OOOOO#\\"+\
        "##O#O#O#O###\\"+\
        "#OO#OOO#OGO#\\"+\
        "############"

LARGE_MAZE_EVAL = \
        "############\\"+\
        "#OO#OOO#OGO#\\"+\
        "##O###O#O#O#\\"+\
        "#OO#O#OOOOO#\\"+\
        "#O##O#OO##O#\\"+\
        "#OOOOOO#OOO#\\"+\
        "#O##O#O#O###\\"+\
        "#OOOO#OOOOO#\\"+\
        "############"

MEDIUM_MAZE = \
        '########\\'+\
        '#OO##OO#\\'+\
        '#OO#OOO#\\'+\
        '##OOO###\\'+\
        '#OO#OOO#\\'+\
        '#O#OO#O#\\'+\
        '#OOO#OG#\\'+\
        "########"

MEDIUM_MAZE_EVAL = \
        '########\\'+\
        '#OOOOOG#\\'+\
        '#O#O##O#\\'+\
        '#OOOO#O#\\'+\
        '###OO###\\'+\
        '#OOOOOO#\\'+\
        '#OO##OO#\\'+\
        "########"

SMALL_MAZE = \
        "######\\"+\
        "#OOOO#\\"+\
        "#O##O#\\"+\
        "#OOOO#\\"+\
        "######"

U_MAZE = \
        "#####\\"+\
        "#GOO#\\"+\
        "###O#\\"+\
        "#OOO#\\"+\
        "#####"

U_MAZE_EVAL = \
        "#####\\"+\
        "#OOG#\\"+\
        "#O###\\"+\
        "#OOO#\\"+\
        "#####"

OPEN = \
        "#######\\"+\
        "#OOOOO#\\"+\
        "#OOGOO#\\"+\
        "#OOOOO#\\"+\
        "#######"

U_MAZE2 = \
        "#####\\"+\
        "#GOO#\\"+\
        "###O#\\"+\
        "#OOO#\\"+\
        "#####"

U_MAZE2_MIRROR = \
        "#####\\"+\
        "#OOG#\\"+\
        "#O###\\"+\
        "#OOO#\\"+\
        "#####"

FUNNEL = \
        '#######\\'+\
        '#OOOOO#\\'+\
        '#OOOOO#\\'+\
        '#OOOOO#\\'+\
        '#OOOOO#\\'+\
        '###G###\\'+\
        "#######"

FUNNEL2 = \
        '#######\\'+\
        '#OOOOO#\\'+\
        '#OOOOO#\\'+\
        '#OOOOO#\\'+\
        '#OOOOO#\\'+\
        '#OOOOO#\\'+\
        '###G###\\'+\
        "#######"

FUNNEL_MULTI_GOAL = \
        '#########\\'+\
        '####G####\\'+\
        '##OOOOO##\\'+\
        '##OOOOO##\\'+\
        '#GOOOOOG#\\'+\
        '##OOOOO##\\'+\
        '##OOOOO##\\'+\
        '####G####\\'+\
        "#########"

SIMPLE_MULTI_GOAL = \
        '#########\\'+\
        '#########\\'+\
        '##OOOOO##\\'+\
        '##OOOOO##\\'+\
        '##OOOOO##\\'+\
        '##OOOOO##\\'+\
        '##OOOOO##\\'+\
        '##G#G#G##\\'+\
        "#########"

SIMPLE_TWO_GOALS = \
        '#########\\'+\
        '#########\\'+\
        '##OOOOO##\\'+\
        '##OOOOO##\\'+\
        '##OOOOO##\\'+\
        '##OOOOO##\\'+\
        '##OOOOO##\\'+\
        '##G###G##\\'+\
        "#########"

SIMPLE_TWO_GOALS2 = \
        '#########\\'+\
        '#########\\'+\
        '##OOOOO##\\'+\
        '##OOOOO##\\'+\
        '##OOOOO##\\'+\
        '##OOOOO##\\'+\
        '##OOOOO##\\'+\
        '###G#G###\\'+\
        "#########"

SIMPLE_TWO_GOALS3 = \
        '#########\\'+\
        '#########\\'+\
        '#OOOOOOO#\\'+\
        '#OOOOOOO#\\'+\
        '#OOOOOOO#\\'+\
        '#OOOOOOO#\\'+\
        '#O#O#O#O#\\'+\
        '###G#G###\\'+\
        "#########"

SIMPLE_CHOKE = \
        '#########\\'+\
        '#########\\'+\
        '##OOOOO##\\'+\
        '##OOOOO##\\'+\
        '###O#####\\'+\
        '##OOOOO##\\'+\
        '##OOOOO##\\'+\
        '#####G###\\'+\
        "#########"


class MazeEnv(mujoco_env.MujocoEnv, utils.EzPickle, offline_env.OfflineEnv):
    def __init__(self,
                 maze_spec=U_MAZE,
                 reward_type='dense',
                 reset_target=False,
                 terminate_at_goal=False,
                 **kwargs):
        offline_env.OfflineEnv.__init__(self, **kwargs)

        self.reset_target = reset_target
        self.str_maze_spec = maze_spec
        self.maze_arr = parse_maze(maze_spec)
        self.reward_type = reward_type
        self.reset_locations = list(zip(*np.where(self.maze_arr == EMPTY)))
        self.reset_locations.sort()
        self.terminate_at_goal = terminate_at_goal

        self._target = np.array([-100.0, -100.0])  # Make sure the initial pos never acheive the goal

        model = point_maze(maze_spec)
        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, model_path=f.name, frame_skip=1)
        utils.EzPickle.__init__(self)

        # Set the default goal (overriden by a call to set_target)
        # Try to find a goal if it exists
        self.goal_locations = list(zip(*np.where(self.maze_arr == GOAL)))
        if len(self.goal_locations) == 1:
            self.set_target(self.goal_locations[0])
        elif len(self.goal_locations) > 1:
            raise ValueError("More than 1 goal specified!")
        else:
            # If no goal, use the first empty tile
            self.set_target(np.array(self.reset_locations[0]).astype(self.observation_space.dtype))
        self.empty_and_goal_locations = self.reset_locations + self.goal_locations

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.clip_velocity()
        self.do_simulation(action, self.frame_skip)
        self.set_marker()
        ob = self._get_obs()
        info = {}
        if self.reward_type == 'sparse':
            reward = 1.0 if np.linalg.norm(ob[0:2] - self._target) <= 0.5 else 0.0
        elif self.reward_type == 'dense':
            reward = np.exp(-np.linalg.norm(ob[0:2] - self._target))
        else:
            raise ValueError('Unknown reward type %s' % self.reward_type)
        done = False

        pos = ob[0:2]
        goal_threshold = 0.5
        if self.terminate_at_goal:
            done = done or (np.linalg.norm(pos - self._target) <= goal_threshold)

        info = {'target': self._target, 'dist_to_target': np.linalg.norm(pos - self._target), **info}

        return ob, reward, done, info

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def get_target(self):
        return self._target

    def set_target(self, target_location=None):
        if target_location is None:
            idx = self.np_random.choice(len(self.empty_and_goal_locations))
            reset_location = np.array(self.empty_and_goal_locations[idx]).astype(self.observation_space.dtype)
            target_location = reset_location + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        self._target = target_location

    def set_marker(self):
        self.data.site_xpos[self.model.site_name2id('target_site')] = np.array([self._target[0]+1, self._target[1]+1, 0.0])

    def clip_velocity(self):
        qvel = np.clip(self.sim.data.qvel, -5.0, 5.0)
        self.set_state(self.sim.data.qpos, qvel)

    def reset_model(self):
        idx = self.np_random.choice(len(self.empty_and_goal_locations))
        reset_location = np.array(self.empty_and_goal_locations[idx]).astype(self.observation_space.dtype)
        qpos = reset_location + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        if self.reset_target:
            self.set_target()
        return self._get_obs()

    def reset_to_location(self, location):
        self.sim.reset()
        reset_location = np.array(location).astype(self.observation_space.dtype)
        qpos = reset_location + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        pass


class FunnelGoalMazeEnv(MazeEnv):
    goal_locs = {'north': (1, 4), 'east': (4, 7), 'south': (7, 4), 'west': (4, 1)}
    def __init__(self,
                 maze_spec=FUNNEL_MULTI_GOAL,
                 reward_type='dense',
                 reset_target=False,
                 goal='south',
                 terminate_at_goal=False,
                 terminate_at_any_goal=False,
                 start_loc=None,
                 bigger_goal=False,
                 **kwargs):
        offline_env.OfflineEnv.__init__(self, **kwargs)

        print('Goal is set to', goal)

        assert not reset_target
        assert goal in self.goal_locs.keys()

        self.bigger_goal = bigger_goal
        self._goal = goal
        self.reset_target = False
        self.str_maze_spec = maze_spec
        self.maze_arr = parse_maze(maze_spec)
        self.reward_type = reward_type

        self.start_loc = start_loc
        if start_loc is not None:
            self.reset_locations = [start_loc]
        else:
            self.reset_locations = list(zip(*np.where(self.maze_arr == EMPTY)))
            self.reset_locations.sort()


        self.terminate_at_goal = terminate_at_goal
        self.terminate_at_any_goal = terminate_at_any_goal

        # NOTE: self._target is referred inside mujoco_env.MujocoEnv.__init__
        # as it executes step(action) internally, which checks the diff between current pos and self._target.
        # Due to this, if self._target is close to ([0., 0.]) it immediately exits.
        self._target = np.array([-999.,-999.])
        model = point_maze(maze_spec)
        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, model_path=f.name, frame_skip=1)
        utils.EzPickle.__init__(self)

        # Set the default goal (overriden by a call to set_target)
        # Try to find a goal if it exists
        self.goal_locations = list(zip(*np.where(self.maze_arr == GOAL)))
        assert self.goal_locs[goal] in self.goal_locations

        # If no goal, use the first empty tile
        self.set_target(np.array(self.goal_locs[goal]).astype(self.observation_space.dtype))

        # HACK: remove goal from this, as the initial location is sampled from this list
        self.empty_and_goal_locations = self.reset_locations

    def step(self, action):
        goal_threshold = 0.5
        offset = np.array([0., 0.])
        if self.bigger_goal:
            goal_threshold += 0.2
            offset = np.array([0.2, 0.])

        obs, rew, done, info = super().step(action)
        pos = obs[0:2]
        if self.terminate_at_goal:
            info['target_reached'] = self._goal
            done = done or (np.linalg.norm(pos - (self._target + offset)) <= goal_threshold)

        if self.terminate_at_any_goal:
            for goal_name, goal_loc in self.goal_locs.items():
                goal_reached = (np.linalg.norm(pos - (goal_loc + offset)) <= goal_threshold)
                if goal_reached:
                    info['target_reached'] = goal_name
                done = done or goal_reached

        return obs, rew, done, info


class SimpleMultiGoalMazeEnv(FunnelGoalMazeEnv):
    goal_locs = {'left': (7, 2), 'center': (7, 4), 'right': (7, 6)}
    def __init__(self, maze_spec=SIMPLE_MULTI_GOAL, reward_type='dense', reset_target=False, goal='center', terminate_at_goal=False, terminate_at_any_goal=False, **kwargs):
        super().__init__(maze_spec=maze_spec, reward_type=reward_type, reset_target=reset_target, goal=goal, terminate_at_goal=terminate_at_goal, terminate_at_any_goal=terminate_at_any_goal, start_loc=(2, 4), **kwargs)

class SimpleTwoGoalsMazeEnv(FunnelGoalMazeEnv):
    """maze2d-simple-two-goals-v0"""
    goal_locs = {'left': (7, 2), 'right': (7, 6)}
    def __init__(self, maze_spec=SIMPLE_TWO_GOALS, reward_type='dense', reset_target=False, goal='left', terminate_at_goal=False, terminate_at_any_goal=False, **kwargs):
        super().__init__(maze_spec=maze_spec, reward_type=reward_type, reset_target=reset_target, goal=goal, terminate_at_goal=terminate_at_goal, terminate_at_any_goal=terminate_at_any_goal, start_loc=None, **kwargs)

class SimpleTwoGoalsMazeEnv2(SimpleTwoGoalsMazeEnv):
    """maze2d-simple-two-goals-v1"""
    goal_locs = {'left': (7, 3), 'right': (7, 5)}
    def __init__(self, maze_spec=SIMPLE_TWO_GOALS2, **kwargs):
        super().__init__(maze_spec=maze_spec, **kwargs)

class SimpleTwoGoalsMazeEnv3(SimpleTwoGoalsMazeEnv):
    """maze2d-simple-two-goals-v2"""
    goal_locs = {'left': (7, 3), 'right': (7, 5)}
    def __init__(self, maze_spec=SIMPLE_TWO_GOALS3, **kwargs):
        super().__init__(maze_spec=maze_spec, **kwargs)


