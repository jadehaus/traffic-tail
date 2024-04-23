import os
import pickle
from argparse import ArgumentParser

from tqdm import tqdm
from linear_rl.true_online_sarsa import TrueOnlineSarsaLambda
from traffic_tail.environment import create_env


class SUMOTrainer(object):
    """
    Main training code.
    Train a DQN model for each module in the environment.
    """
    def __init__(self, env='default', num_seconds=7200, use_gui=False):
        self.result_dir = f"results/{env}"
        self.best_reward = -float('inf')
        
        if env == 'default':
            tailgating = False
            default_mode = 31
        elif env == 'tailgating':
            tailgating = True
            default_mode = 31
        elif env == 'overspeed':
            tailgating = False
            default_mode = 24
        elif env == 'impatience':
            tailgating = True
            default_mode = 24
        else:
            raise ValueError(f"Invalid environment {env}")
        
        self.env = create_env(
            tailgating=tailgating,
            default_mode=default_mode,
            use_gui=use_gui,
            num_seconds=num_seconds,
        )
        
        print(f"Initializing RL agents. (This may take a while)")
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
        pbar = tqdm(range(episodes * self.env.sim_max_time))
        for episode in range(episodes):
            total_reward = 0
            state = self.env.reset()
            done = {"__all__": False}
            
            pbar.set_description(f"Episode {episode}/{episodes}: Total Reward --")
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
                
                total_reward += sum(reward.values())
                pbar.update(self.env.delta_time)
            
            # self.save(f'{self.result_dir}/pretrained_agents_run_{episode}.pkl')
            pbar.set_description(
                f"Episode {episode}/{episodes}: Total Reward {total_reward:.3f}"
            )
            
            if total_reward > self.best_reward:
                self.best_reward = total_reward
                self.save(f'{self.result_dir}/best_agents.pkl')
            
        self.env.close()
        return self.agents
    
    def save(self, path=None):
        if path is None:
            path = os.path.join(
                self.result_dir, 
                'pretrained_agents.pkl'
            )
        with open(path, 'wb') as f:
            pickle.dump(self.agents, f)
            
    def load(self, path):
        with open(path, 'rb') as f:
            self.agents = pickle.load(f)
        return self


parser = ArgumentParser()
parser.add_argument('--use-gui', action='store_true', default=False)
parser.add_argument('--env', type=str, default='default')


if __name__ == "__main__":
    args = parser.parse_args()
    trainer = SUMOTrainer(env=args.env)
    trainer.train()