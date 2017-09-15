import gym

from baselines import deepq
import Traci_1_cross_env.Traci_1_cross_env
import Traci_2_cross_env.Traci_2_cross_env
from baselines import logger, logger_utils


def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved

def main():

    env = gym.make('Traci_2_cross_env-v0')
    env.configure_traci(steps=500)
    #env.render()

    logger.reset()
    logger_path = logger_utils.path_with_date("/tmp/Traci_2_cross_env-v0", "Traci_2_cross_env-v0")
    logger.configure(logger_path, ["tensorboard", "stdout"])
    model = deepq.models.mlp([64])
    act = deepq.learn(
        env,
        q_func=model,
        callback=None,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        checkpoint_freq=10000,
        exploration_fraction=0.4,
        gamma=0.9,
        exploration_final_eps=0.02,
        print_freq=1,
        print_timestep_freq=100,
        log_path="/tmp/traci",
        model_path="traffic_model.pkl",
        num_cpu=4,
        prioritized_replay=True
    )
    print("Saving model to traffic_model.pkl")
    act.save("traffic_model.pkl")


if __name__ == '__main__':
    main()
