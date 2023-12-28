import sys

from sample_factory.algo.utils.rl_utils import make_dones
from sample_factory.envs.create_env import create_env
from sample_factory.utils.utils import log
from sf_examples.nethack.train_nethack import parse_nethack_args, register_nethack_components


def main():
    register_nethack_components()
    cfg = parse_nethack_args(evaluation=True)

    render_mode = "human" if not cfg.no_render else None
    env = create_env(cfg.env, cfg=cfg, render_mode=render_mode)

    for i in range(10):
        env.reset()
        done = False
        while not done:
            obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
            done = make_dones(terminated, truncated)
    log.info("Done!")


if __name__ == "__main__":
    sys.exit(main())
