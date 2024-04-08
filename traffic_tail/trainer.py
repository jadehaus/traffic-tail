import os
from argparse import ArgumentParser

from linear_rl.true_online_sarsa import TrueOnlineSarsaLambda
from traffic_tail.environment import TailGatingEnv


class Trainer(object):
    """
    Main training code.
    Train a DQN model for each module in the environment.
    """
    def __init__(self, env='default'):
        self.result_dir = f"results/{env}"
        
        if env == 'default':
            tailgating = False
        elif env == 'tailgating':
            tailgating = True
        else:
            raise ValueError(f"Invalid environment {env}")
        
        self.env = TailGatingEnv(
            tailgating=tailgating,
            net_file="nets/network.net.xml",
            route_file="nets/flow.rou.xml",
            single_agent=False,
            out_csv_name=self.result_dir,
            use_gui=False,
            num_seconds=86400,
            yellow_time=3,
            min_green=5,
            max_green=60,
            sumo_warnings=False,
        )   
        
        self.agents = {
            ts_id: TrueOnlineSarsaLambda(
                self.env.observation_spaces(ts_id),
                self.env.action_spaces(ts_id),
                alpha=0.000000001,
                gamma=0.95,
                epsilon=0.05,
                lamb=0.1,
                fourier_order=7,
            )
            for ts_id in self.env.ts_ids
        }
    
    def train(self, episodes=1):
        for episode in range(episodes):
            state = self.env.reset()
            done = {"__all__": False}
            while not done["__all__"]:
                actions = {
                    ts_id: self.agents[ts_id].act(state[ts_id]) 
                    for ts_id in state.keys()
                }
                
                next_state, reward, done, _ = self.env.step(action=actions)

                for ts_id in next_state.keys():
                    self.agents[ts_id].learn(
                        state=next_state[ts_id], 
                        action=actions[ts_id], 
                        reward=reward[ts_id], 
                        next_state=next_state[ts_id], 
                        done=done[ts_id]
                    )
                    state[ts_id] = next_state[ts_id]
                
            self.env.save_csv(self.result_dir, episode)
        self.env.close()
        return self.agents


parser = ArgumentParser()
parser.add_argument('--use-gui', action='store_true', default=False)
parser.add_argument('--env', type=str, default='default')


if __name__ == "__main__":
    args = parser.parse_args()
    trainer = Trainer(env=args.env)
    trainer.train()