#!/usr/bin/env python3
import gym
from d4rl.pointmaze import waypoint_controller
from d4rl.pointmaze import maze_model
import numpy as np
import h5py
import argparse

from .generate_maze2d_datasets import reset_data, append_data, npify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Render trajectories')
    parser.add_argument('--noisy', action='store_true', help='Noisy actions')
    parser.add_argument('--env_name', type=str, default='maze2d-funnel-v1', help='Maze type')
    parser.add_argument('--num_samples', type=int, default=int(1e6), help='Num samples to collect')
    args = parser.parse_args()
    assert 'funnel' in args.env_name

    env = gym.make(args.env_name)
    maze = env.str_maze_spec
    max_episode_steps = env._max_episode_steps

    controller = waypoint_controller.WaypointController(maze)
    env = maze_model.MazeEnv(maze, reward_type='sparse')

    def wrapped_reset():
        # env.empty_and_goal_locations is a list of tuple (list of positions)
        # Format is (x, y) where x goes down and y goes to the right
        prev_array = env.empty_and_goal_locations.copy()
        new_array = [loc for loc in env.empty_and_goal_locations if loc[0] < 3]
        env.empty_and_goal_locations = new_array
        obs = env.reset()
        env.empty_and_goal_locations = prev_array
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
        if ts >= max_episode_steps:
            done = True
        append_data(data, s, act, env._target, done, env.sim.data)

        ns, rew, _, _ = env.step(act)

        if len(data['observations']) % 10000 == 0:
            print(len(data['observations']))

        # NOTE: Assuming sparse reward, terminate the episode when non-zero reward is given!
        if rew > 0:
            print('step', i, 'rew', rew)
            data['terminals'][-1] = True  # HACK
            done = True

        ts += 1
        if done:
            s = wrapped_reset()
            done = False
            ts = 0
        else:
            s = ns

        if args.render:
            env.render()


    import os
    directory = '/data/maze2d'
    os.makedirs(directory, exist_ok=True)
    if args.noisy:
        fname = os.path.join(directory, '%s-noisy.hdf5' % args.env_name)
    else:
        fname = os.path.join(directory, '%s.hdf5' % args.env_name)
    dataset = h5py.File(fname, 'w')
    npify(data)
    for k in data:
        dataset.create_dataset(k, data=data[k], compression='gzip')


if __name__ == "__main__":
    main()
