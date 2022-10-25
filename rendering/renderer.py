#!/usr/bin/env python3
import os
import numpy as np
import einops
import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import gym
import warnings
import pdb

# from .arrays import to_np
# from .video import save_video, save_videos

# from diffuser.datasets.d4rl import load_environment


#-----------------------------------------------------------------------------#
#------------------------------ helper functions -----------------------------#
#-----------------------------------------------------------------------------#

def atmost_2d(x):
    while x.ndim > 2:
        x = x.squeeze(0)
    return x

def zipsafe(*args):
    length = len(args[0])
    assert all([len(a) == length for a in args])
    return zip(*args)

def zipkw(*args, **kwargs):
    nargs = len(args)
    keys = kwargs.keys()
    vals = [kwargs[k] for k in keys]
    zipped = zipsafe(*args, *vals)
    for items in zipped:
        zipped_args = items[:nargs]
        zipped_kwargs = {k: v for k, v in zipsafe(keys, items[nargs:])}
        yield zipped_args, zipped_kwargs

def get_image_mask(img):
    background = (img == 255).all(axis=-1, keepdims=True)
    mask = ~background.repeat(3, axis=-1)
    return mask

def plot2img(fig, remove_margins=True):
    # https://stackoverflow.com/a/35362787/2912349
    # https://stackoverflow.com/a/54334430/2912349

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if remove_margins:
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img_as_string, (width, height) = canvas.print_to_buffer()
    return np.fromstring(img_as_string, dtype='uint8').reshape((height, width, 4))

#-----------------------------------------------------------------------------#
#---------------------------------- renderers --------------------------------#
#-----------------------------------------------------------------------------#

MAZE_BOUNDS = {
    'maze2d-umaze2-v0': (0, 5, 0, 5),
    'maze2d-umaze2-mirror-v0': (0, 5, 0, 5),
    'maze2d-umaze-v1': (0, 5, 0, 5),
    'maze2d-medium-v1': (0, 8, 0, 8),
    'maze2d-funnel-v0': (0, 7, 0, 7),
    'maze2d-large-v1': (0, 9, 0, 12)
}

class MazeRenderer:
    def __init__(self, env):
        self._config = env._config
        self._background = self._config != ' '
        self._remove_margins = False
        self._extent = (0, 1, 1, 0)

    def renders(self, observations, title=None, updated_background=None):
        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches(5, 5)
        if updated_background is not None:
            _background = updated_background
        else:
            _background = self._background
        plt.imshow(_background * .5,
            extent=self._extent, cmap=plt.cm.binary, vmin=0, vmax=1)

        path_length = len(observations)
        colors = plt.cm.jet(np.linspace(0,1,path_length))
        plt.plot(observations[:,1], observations[:,0], c='black', zorder=10)
        plt.scatter(observations[:,1], observations[:,0], c=colors, zorder=20)
        plt.axis('off')
        plt.title(title)
        img = plot2img(fig, remove_margins=self._remove_margins)
        return img

    def composite(self, savepath, paths, ncol=5, **kwargs):
        '''
            savepath : str
            observations : [ n_paths x horizon x 2 ]
        '''
        assert len(paths) % ncol == 0, 'Number of paths must be divisible by number of columns'

        images = []
        for path, kw in zipkw(paths, **kwargs):
            img = self.renders(*path, **kw)
            images.append(img)
        images = np.stack(images, axis=0)

        nrow = len(images) // ncol
        images = einops.rearrange(images,
            '(nrow ncol) H W C -> (nrow H) (ncol W) C', nrow=nrow, ncol=ncol)
        if savepath is not None:
            imageio.imsave(savepath, images)
            print(f'Saved {len(paths)} samples to: {savepath}')

        return images

    def single_composite(self, savepath, obs, ncol=1, **kwargs):
        '''
                    savepath : str
                    observations : [ n_paths x horizon x 2 ]
                '''
        assert len(obs) % ncol == 0, 'Number of paths must be divisible by number of columns'

        images = self.renders(obs)
        imageio.imsave(savepath, images)
        print(f'Saved {len(obs)} samples to: {savepath}')


class Maze2dRenderer(MazeRenderer):

    def __init__(self, env, bounds):
        self.env = env
        self.bounds = bounds

        self.observation_dim = np.prod(self.env.observation_space.shape)
        self.action_dim = np.prod(self.env.action_space.shape)
        self.goal = None
        self._background = self.env.maze_arr == 10
        self._remove_margins = False
        self._extent = (0, 1, 1, 0)

    def renders(self, observations, conditions=None, **kwargs):
        observations = observations + .5
        if len(self.bounds) == 2:
            _, scale = self.bounds
            observations /= scale
        elif len(self.bounds) == 4:
            _, iscale, _, jscale = self.bounds
            observations[:, 0] /= iscale
            observations[:, 1] /= jscale
        else:
            raise RuntimeError(f'Unrecognized bounds for {self.env}: {self.bounds}')

        if conditions is not None:
            conditions /= scale
        return super().renders(observations, conditions, **kwargs)

#-----------------------------------------------------------------------------#
#---------------------------------- rollouts ---------------------------------#
#-----------------------------------------------------------------------------#

def set_state(env, state):
    qpos_dim = env.sim.data.qpos.size
    qvel_dim = env.sim.data.qvel.size
    if not state.size == qpos_dim + qvel_dim:
        warnings.warn(
            f'[ utils/rendering ] Expected state of size {qpos_dim + qvel_dim}, '
            f'but got state of size {state.size}')
        state = state[:qpos_dim + qvel_dim]

    env.set_state(state[:qpos_dim], state[qpos_dim:])

def rollouts_from_state(env, state, actions_l):
    rollouts = np.stack([
        rollout_from_state(env, state, actions)
        for actions in actions_l
    ])
    return rollouts

def rollout_from_state(env, state, actions):
    qpos_dim = env.sim.data.qpos.size
    env.set_state(state[:qpos_dim], state[qpos_dim:])
    observations = [env._get_obs()]
    for act in actions:
        obs, rew, term, _ = env.step(act)
        observations.append(obs)
        if term:
            break
    for i in range(len(observations), len(actions)+1):
        ## if terminated early, pad with zeros
        observations.append( np.zeros(obs.size) )
    return np.stack(observations)



def main(env_name, h5path, outdir, num_episodes, min_traj_len=10):
    from d4rl.pointmaze.maze_model import U_MAZE, U_MAZE2, OPEN, MazeEnv
    from PIL import Image
    import os
    from pathlib import Path

    # env = MazeEnv(U_MAZE2)
    env = gym.make(env_name)
    # maze = env.str_maze_spec
    # max_episode_steps = env._max_episode_steps

    # NOTE: dataset is a dict. Keys: dict_keys(['actions', 'infos/goal', 'infos/qpos', 'infos/qvel', 'observations', 'rewards', 'terminals'])
    # dataset['observations'].shape: (500000, 4)
    dataset = env.get_dataset(h5path=h5path)
    renderer = Maze2dRenderer(env, bounds=MAZE_BOUNDS[env_name])

    terminal_steps = np.where(dataset['terminals'])[0]
    print(f'There are {len(terminal_steps)} episodes in the dataset!')
    prev_term_step = 0
    counter = 0
    for term_step in terminal_steps:
        term_step = term_step + 1  # Why??
        if term_step - prev_term_step < min_traj_len:
            print(f'A short trajectory with len {term_step - prev_term_step} < {min_traj_len} is rejected.')
            prev_term_step = term_step
            continue

        ep = dataset['observations'][prev_term_step:term_step]
        img = renderer.renders(ep)  # shape: (500 , 500, 4) (RGBA?)
        Image.fromarray(img).save(Path(outdir) / f'ep_{counter:04d}.png')

        prev_term_step = term_step
        counter += 1

        if counter == num_episodes:
            break

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='maze2d-funnel-v0', help='Maze type')
    parser.add_argument('--h5path', type=str, default='/data/maze2d/maze2d-funnel-v0.hdf5', help='Path to h5py file')
    parser.add_argument('--outdir', type=str, default=os.environ['RMX_OUTPUT_DIR'], help='Where to save the generated figures')
    parser.add_argument('--num-episodes', type=int, default=100, help='number of episodes to visualize')
    args = parser.parse_args()
    main(args.env_name, args.h5path, args.outdir, args.num_episodes)
