import traci
from sumo_rl import SumoEnvironment

'''
Speed Mode
bit0: Regard safe speed
bit1: Regard maximum acceleration
bit2: Regard maximum deceleration
bit3: Regard right of way at intersections (only applies to approaching foe vehicles outside the intersection)
bit4: Brake hard to avoid passing a red light
bit5: Disregard right of way within intersections (only applies to foe vehicles that have entered the intersection).
Setting the bit enables the check (the according value is regarded), keeping the bit==zero disables the check.



Examples:
default (all checks on) -> [0 1 1 1 1 1] -> Speed Mode = 31
most checks off (legacy) -> [0 0 0 0 0 0] -> Speed Mode = 0
all checks off -> [1 0 0 0 0 0] -> Speed Mode = 32
disable right of way check -> [1 1 0 1 1 1] -> Speed Mode = 55
run a red light [0 0 0 1 1 1] = 7 (also requires setSpeed or slowDown)
run a red light even if the intersection is occupied [1 0 0 1 1 1] = 39 (also requires setSpeed or slowDown)
'''


class TailGatingEnv(SumoEnvironment):
    def __init__(self, tailgating=True, default_mode=31, *args, **kwargs):
        super(TailGatingEnv, self).__init__(*args, **kwargs)
        self.tailgating = tailgating
        self.default_mode = default_mode

    def _set_default_mode(self):
        for vehID in self.sumo.vehicle.getIDList():
            self.sumo.vehicle.setSpeedMode(vehID, self.default_mode)
        print(self.sumo.vehicle.getIDList())
                        
    def _apply_tailgating(self):
        for tlsID in self.sumo.trafficlight.getIDList():
            controlledLanes = self.sumo.trafficlight.getControlledLanes(tlsID)
            stateString = self.sumo.trafficlight.getRedYellowGreenState(tlsID)
            
            for idx, lane in enumerate(controlledLanes):
                vehicles = self.sumo.lane.getLastStepVehicleIDs(lane)
                for vehID in vehicles:
                    if 'y' in stateString[idx]:
                        self.sumo.vehicle.setSpeedMode(vehID, 0)
                    
            
    def _apply_realistic_impatience_gap(self):
        for vehID in self.sumo.vehicle.getIDList():
            impatience = self.sumo.vehicle.getImpatience(vehID)
            minGap = 2.5 if impatience < 1 else 0.5
            self.sumo.vehicle.setMinGap(vehID, minGap)

    def _sumo_step(self):
        self._set_default_mode()
        if self.tailgating:
            self._apply_tailgating()
            # self._apply_realistic_impatience_gap()
        self.sumo.simulationStep()
        
        
        

def create_env(env_config):
    print(f"Creating {env_config.name} environment.")
    if env_config.tailgating:
        net_file = "/mnt/d/Working_/GitHub/traffic-tail/nets/network.net.xml"
        route_file = "/mnt/d/Working_/GitHub/traffic-tail/nets/flow_tailgating.rou.xml"
    else:
        net_file = "/mnt/d/Working_/GitHub/traffic-tail/nets/network.net.xml"
        route_file = "/mnt/d/Working_/GitHub/traffic-tail/nets/flow_default.rou.xml"
    return TailGatingEnv(
        tailgating=env_config.tailgating,
        default_mode=env_config.default_mode,
        net_file=net_file,
        route_file=route_file,
        single_agent=False,
        use_gui=env_config.use_gui,
        num_seconds=env_config.num_seconds,
        yellow_time=3,
        min_green=5,
        max_green=60,
        delta_time=5,
        sumo_warnings=False,
    )