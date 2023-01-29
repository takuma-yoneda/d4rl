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
    parser.add_argument('--envname', default='maze2d-simple-two-goals-v0', help='env name')
    parser.add_argument('--render', action='store_true', help='Render trajectories')
    parser.add_argument('--noisy', action='store_true', help='Noisy actions')
    parser.add_argument('--num_samples', type=int, default=int(1e6), help='Num samples to collect')
    args = parser.parse_args()

    env_name = args.envname
    env = gym.make(env_name)
    maze = env.str_maze_spec
    max_episode_steps = env._max_episode_steps

    controller = waypoint_controller.WaypointController(maze)
    init_pos_noise = 0.4

    for goal in ['left', 'right']:
        print('goal', goal)
        env = gym.make(env_name, goal=goal, reward_type='sparse', terminate_at_goal=True, init_pos_noise=init_pos_noise)
        # env = maze_model.SimpleTwoGoalsMazeEnv(goal=goal, reward_type='sparse', terminate_at_goal=True)
        assert env.terminate_at_goal and not env.terminate_at_any_goal

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
        done = False

        data = reset_data()
        ts = 0
        counter = 0
        while counter < args.num_samples:
            position = s[0:2]
            velocity = s[2:4]
            act, done = controller.get_action(position, velocity, env.unwrapped._target)
            if args.noisy:
                act = act + np.random.randn(*act.shape)*0.5

            act = np.clip(act, -1.0, 1.0)
            append_data(data, s, act, env.unwrapped._target, done, env.sim.data)

            ns, rew, done, info = env.step(act)

            if ts >= max_episode_steps:
                done = True

            if len(data['observations']) % 10000 == 0:
                print(len(data['observations']))

            ts += 1
            if done:
                data['terminals'][-1] = True  # HACK
                if info.get('target_reached', '') != goal:
                    print('warn: target not reached! Rejecting the trajectory...')
                    # Goal is not reached for whatever reason!!
                    # --> Remove the corresponding number of transitions!!
                    counter -= ts
                    for key in data:
                        data[key] = data[key][:-ts]

                s = wrapped_reset()
                done = False
                ts = 0
            else:
                s = ns

            if args.render:
                env.render()

            counter += 1


        if args.noisy:
            fname = os.path.join(directory, f'%s-%s-noisy-initpos{init_pos_noise:.1f}.hdf5' % (env_name, goal))
        else:
            fname = os.path.join(directory, f'%s-%s-initpos{init_pos_noise:.1f}.hdf5' % (env_name, goal))
        dataset = h5py.File(fname, 'w')
        npify(data)
        for k in data:
            print(f'key: {k}\t length: {len(data[k])}')
            dataset.create_dataset(k, data=data[k], compression='gzip')


if __name__ == "__main__":
    import os
    directory = '/data/maze2d-initloc'
    os.makedirs(directory, exist_ok=True)
    main(directory)
