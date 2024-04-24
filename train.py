import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns

from tqdm import tqdm
from traffic_tail.environment import create_env
from traffic_tail.trainer import SUMOTrainer


USE_SUMO_GUI = False
TOTAL_TIME = 900
NUM_SEEDS = 5
NUM_EPISODES = 20


class DefaultConfig:
    name = "default"
    use_gui = USE_SUMO_GUI
    num_seconds = TOTAL_TIME
    tailgating = False
    default_mode = 31


class OverspeedConfig:
    name = "overspeed"
    use_gui = USE_SUMO_GUI
    num_seconds = TOTAL_TIME
    tailgating = False
    default_mode = 24
    

class TailgatingConfig:
    name = "tailgating"
    use_gui = USE_SUMO_GUI
    num_seconds = TOTAL_TIME
    tailgating = True
    default_mode = 31
    

class TailgatingOverspeedConfig:
    name = "tailgating_overspeed"
    use_gui = USE_SUMO_GUI
    num_seconds = TOTAL_TIME
    tailgating = True
    default_mode = 24
    

def run_experiment(config):
    reward_curve = []
    for seed in range(NUM_SEEDS):
        trainer_default = SUMOTrainer(config)
        trainer_default.train(episodes=NUM_EPISODES, run=seed)
        reward_curve.append(trainer_default.total_rewards)
        
    reward_curve = np.array(reward_curve)
    np.save(f"results/rewards_{config.name}.npy", reward_curve)


if __name__ == '__main__':
    
    default_config = DefaultConfig()
    overspeed_config = OverspeedConfig()
    tailgating_config = TailgatingConfig()
    tailgating_overspeed_config = TailgatingOverspeedConfig()
    
    for config in [default_config, overspeed_config, tailgating_config, tailgating_overspeed_config]:
        run_experiment(config)