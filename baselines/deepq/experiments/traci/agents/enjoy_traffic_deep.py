import gym

from baselines import deepq
import Traci_2_cross_env.Traci_2_cross_env

def main():
    env = gym.make('Traci_2_cross_env-v0')
    act = deepq.load("traffic_model.pkl")
    env.render()

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:

            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
