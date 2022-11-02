#!/usr/bin/env python3
import gym
from d4rl.pointmaze import waypoint_controller
from d4rl.pointmaze import maze_model
import numpy as np
import h5py
import argparse

from .generate_maze2d_datasets import reset_data, append_data, npify


def main(directory):
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Render trajectories')
    parser.add_argument('--noisy', action='store_true', help='Noisy actions')
    parser.add_argument('--num_samples', type=int, default=int(1e6), help='Num samples to collect')
    args = parser.parse_args()

    env_name = 'maze2d-funnel-goal-v0'
    env = gym.make(env_name)
    maze = env.str_maze_spec
    max_episode_steps = env._max_episode_steps

    controller = waypoint_controller.WaypointController(maze)

    for goal in ['north', 'east', 'south', 'west']:
        print('goal', goal)
        env = maze_model.FunnelGoalMazeEnv(goal=goal, reward_type='sparse', terminate_at_goal=True)

        def wrapped_reset():
            # env.empty_and_goal_locations is a list of tuple (list of positions)
            # Format is (x, y) where x goes down and y goes to the right
            # prev_array = env.empty_and_goal_locations.copy()
            # new_array = [loc for loc in env.empty_and_goal_locations if loc[0] < 3]
            # env.empty_and_goal_locations = new_array
            obs = env.reset()
            # env.empty_and_goal_locations = prev_array
            return obs

        s = wrapped_reset()
        act = env.action_space.sample()
        done = False

        data = reset_data()
        ts = 0
        for i in range(args.num_samples):
            position = s[0:2]
            velocity = s[2:4]
            act, done = controller.get_action(position, velocity, env._target)
            if args.noisy:
                act = act + np.random.randn(*act.shape)*0.5

            act = np.clip(act, -1.0, 1.0)
            append_data(data, s, act, env._target, done, env.sim.data)

            ns, rew, done, _ = env.step(act)

            if ts >= max_episode_steps:
                done = True

            if len(data['observations']) % 10000 == 0:
                print(len(data['observations']))

            ts += 1
            if done:
                data['terminals'][-1] = True  # HACK
                s = wrapped_reset()
                done = False
                ts = 0
            else:
                s = ns

            if args.render:
                env.render()


        if args.noisy:
            fname = os.path.join(directory, '%s-%s-noisy.hdf5' % (env_name, goal))
        else:
            fname = os.path.join(directory, '%s-%s.hdf5' % (env_name, goal))
        dataset = h5py.File(fname, 'w')
        npify(data)
        for k in data:
            dataset.create_dataset(k, data=data[k], compression='gzip')


if __name__ == "__main__":
    import os
    directory = '/data/maze2d'
    os.makedirs(directory, exist_ok=True)
    main(directory)
