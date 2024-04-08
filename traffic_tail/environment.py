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
    def __init__(self, tailgating=True, *args, **kwargs):
        super(TailGatingEnv, self).__init__(*args, **kwargs)
        self.tailgating = tailgating
                        
    def _adjust_speed_mode(self, default_mode=31):
        for tlsID in self.sumo.trafficlight.getIDList():
            controlledLanes = self.sumo.trafficlight.getControlledLanes(tlsID)
            stateString = self.sumo.trafficlight.getRedYellowGreenState(tlsID)
            
            for idx, lane in enumerate(controlledLanes):
                vehicles = self.sumo.lane.getLastStepVehicleIDs(lane)
                for vehID in vehicles:
                    if 'y' in stateString[idx]:
                        self.sumo.vehicle.setSpeedMode(vehID, 0)
                    elif 'G' in stateString[idx]:
                        self.sumo.vehicle.setSpeedMode(vehID, 7)
                    else:
                        self.sumo.vehicle.setSpeedMode(vehID, default_mode)

    def _sumo_step(self):
        if self.tailgating:
            self._adjust_speed_mode()
        self.sumo.simulationStep()
        

def create_env(tailgating=False, use_gui=False):
    return TailGatingEnv(
        tailgating=tailgating,
        net_file="nets/network.net.xml",
        route_file="nets/flow.rou.xml",
        single_agent=False,
        use_gui=use_gui,
        num_seconds=86400,
        yellow_time=3,
        min_green=5,
        max_green=60,
        sumo_warnings=False,
    )