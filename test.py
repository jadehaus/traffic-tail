import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns

from glob import glob
from tqdm import tqdm
from traffic_tail.environment import create_env
from traffic_tail.trainer import SUMOTrainer


USE_SUMO_GUI = False
TOTAL_TIME = 900
NUM_SEEDS = 5
NUM_EPISODES = 40


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


def run_episode(env, agent=None):
    total_reward = 0
    state = env.reset()
    done = {"__all__": False}
    while not done["__all__"]:
        if agent is None:
            actions = {
                ts_id: env.action_spaces(ts_id).sample()
                for ts_id in env.ts_ids
            }
        else:
            actions = {
                ts_id: agent[ts_id].act(state[ts_id]) 
                for ts_id in state.keys()
            }
        state, reward, done, _ = env.step(actions)
        total_reward += sum(reward.values())
    env.close()
    return total_reward


if __name__ == "__main__":
    default_config = DefaultConfig()
    overspeed_config = OverspeedConfig()
    tailgating_config = TailgatingConfig()
    tailgating_overspeed_config = TailgatingOverspeedConfig()

    default_env = create_env(default_config)
    tailgating_env = create_env(tailgating_config)
    overspeeding_env = create_env(overspeed_config)
    tailgating_overspeed_env = create_env(tailgating_overspeed_config)

    _ddr = []
    _dtr = []
    _tdr = []
    _ttr = []
    _oor = []
    _odr = []
    _dor = []
    _idr = []
    _iir = []
    _dir = []

    _dr = []
    _tr = []
    _or = []
    _ir = []

    for seed in tqdm(range(NUM_SEEDS)):
        _dr.append(run_episode(default_env))
        _tr.append(run_episode(tailgating_env))
        _or.append(run_episode(overspeeding_env))
        _ir.append(run_episode(tailgating_overspeed_env))

        trainer_default = SUMOTrainer(default_config).load(f'results/default/best_agents_run_{seed}.pkl')
        trainer_tailgating = SUMOTrainer(tailgating_config).load(f'results/tailgating/best_agents_run_{seed}.pkl')
        trainer_overspeeding = SUMOTrainer(overspeed_config).load(f'results/overspeed/best_agents_run_{seed}.pkl')
        trainer_tailgating_overspeed = SUMOTrainer(tailgating_overspeed_config).load(f'results/tailgating_overspeed/best_agents_run_{seed}.pkl')
        default_agent = trainer_default.agents
        tailgating_agent = trainer_tailgating.agents
        overspeeding_agent = trainer_overspeeding.agents
        tailgating_overspeed_agent = trainer_tailgating_overspeed.agents

        _ddr.append(run_episode(default_env, default_agent))
        _ttr.append(run_episode(tailgating_env, tailgating_agent))
        _tdr.append(run_episode(tailgating_env, default_agent))
        _dtr.append(run_episode(default_env, tailgating_agent))
        _oor.append(run_episode(overspeeding_env, overspeeding_agent))
        _odr.append(run_episode(overspeeding_env, default_agent))
        _dor.append(run_episode(default_env, overspeeding_agent))
        _idr.append(run_episode(tailgating_overspeed_env, default_agent))
        _iir.append(run_episode(tailgating_overspeed_env, tailgating_overspeed_agent))
        _dir.append(run_episode(default_env, tailgating_overspeed_agent))



    print(f"Default Agent in Default Environment: {sum(_ddr)/len(_ddr)}")
    print(f"Tailgating Agent in Tailgating Environment: {sum(_ttr)/len(_ttr)}")
    print(f"Default Agent in Tailgating Environment: {sum(_tdr)/len(_tdr)}")
    print(f"Tailgating Agent in Default Environment: {sum(_dtr)/len(_dtr)}")
    print(f"Overspeeding Agent in Overspeeding Environment: {sum(_oor)/len(_oor)}")
    print(f"Overspeeding Agent in Default Environment: {sum(_dor)/len(_dor)}")
    print(f"Default Agent in Overspeeding Environment: {sum(_odr)/len(_odr)}")
    print(f"Default Agent in Tailgating-Overspeeding Environment: {sum(_idr)/len(_idr)}")
    print(f"Tailgating-Overspeeding Agent in Tailgating-Overspeeding Environment: {sum(_iir)/len(_iir)}")
    print(f"Tailgating-Overspeeding Agent in Default Environment: {sum(_dir)/len(_dir)}")

    print(f"Random Agent in Default Environment: {sum(_dr)/len(_dr)}")
    print(f"Random Agent in Tailgating Environment: {sum(_tr)/len(_tr)}")
    print(f"Random Agent in Overspeeding Environment: {sum(_or)/len(_or)}")
    print(f"Random Agent in Tailgating-Overspeeding Environment: {sum(_ir)/len(_ir)}")