#!/usr/bin/env python3
from pathlib import Path
import gym
from d4rl.pointmaze import waypoint_controller
from d4rl.pointmaze import maze_model
import numpy as np
import h5py
import argparse

from .generate_maze2d_datasets import reset_data, append_data, npify

def save_video(frame_stack, path, fps=20, **imageio_kwargs):
    """Save a vidoe from a list of frames.

    Correspondence: https://github.com/geyang/ml_logger
    """
    import os
    import tempfile, imageio  # , logging as py_logging
    import shutil
    # py_logging.getLogger("imageio").setLevel(py_logging.WARNING)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    format = 'mp4'
    with tempfile.NamedTemporaryFile(suffix=f'.{format}') as ntp:
        from skimage import img_as_ubyte
        try:
            imageio.mimsave(ntp.name, img_as_ubyte(frame_stack), format=format, fps=fps, **imageio_kwargs)
        except imageio.core.NeedDownloadError:
            imageio.plugins.ffmpeg.download()
            imageio.mimsave(ntp.name, img_as_ubyte(frame_stack), format=format, fps=fps, **imageio_kwargs)
        ntp.seek(0)
        shutil.copy(ntp.name, path)


def main(args):

    env_name = args.envname
    env = gym.make(env_name)
    init_pos_noise = 0.4

    for goal in ['left', 'right']:
        print('goal', goal)
        env = gym.make(env_name, goal=goal, reward_type='sparse', terminate_at_goal=True, init_pos_noise=init_pos_noise)

        frames = []
        for i in range(args.num_episodes):
            obs = env.reset()
            img = env.render(mode='rgb_array')
            frames.append(img)

        save_video(frames, path=Path(args.outdir) / 'frames.mp4', fps=3)


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--envname', default='maze2d-simple-two-goals-v0', help='env name')
    parser.add_argument('--num-episodes', type=int, default=20, help='number of episodes')
    parser.add_argument('--outdir', default=os.environ['RMX_OUTPUT_DIR'], help='output directory')
    args = parser.parse_args()

    main(args)
