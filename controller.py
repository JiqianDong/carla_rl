import carla
import pickle
import numpy as np

class controller(object):
    def __init__(self,  actor, carla_pilot=False):

        self.vehilcle = actor
        self.carla_pilot = carla_pilot
        
        self.current_control = {'throttle':0,"steer":0,"brake":0}

        self.steering_increment = 0.1

        self.reset()

        if self.carla_pilot:
            self.vehilcle.set_autopilot()
        else:
            self._control = carla.VehicleControl()
        
    def reset(self):
        self.timestep = 0

    def step(self, decisions=None):
        self.timestep += 1
        if isinstance(decisions,np.ndarray):
            dec = process(decisions)
            self.apply(dec) 
        

    def process(self, dec):
        dec = decisions.copy()
        return dec

    def apply(self, dec):
        # update current booking 
        self.current_control['throttle'] = dec[0]
        self.current_control['steer'] = dec[1]
        self.current_control['brake'] = dec[2]
        
        self._control.throttle = self.current_control['throttle']
        self._control.steer = self.current_control['steer']
        self._control.brake = self.current_control['brake']

        self.vehilcle.apply_control(self._control)

        

class CAV_controller(controller):
    def process(self,dec):
        dec -= 1

        if dec[0]==-1:
            throttle = 0
            brake = 1
        elif dec[0] == 0:
            throttle = 0
            brake = 0
        elif dec[0]==1:
            throttle = 1
            brake = 0
        else:
            print(dec)
            raise Exception("no specific throttle brake control command")

        steer = self.current_control['steer'] + dec[1]*self.steering_increment

        return [throttle, steer, brake]


class LHDV_controller(controller):
    def __init__(self, actor, carla_pilot=False, command_file='./control_details/LHDV.p'):
        super().__init__(actor, carla_pilot)
        with open(command_file,'rb') as f:
            self.command_list = pickle.load(f)

    def process(self, dec):
        throttle = dec['throttle']
        steer = dec['steer']
        brake = dec['brake']
        return [throttle, steer, brake]

    def step(self,decisions=None):
        self.timestep += 1
        decisions = self.command_list[self.timestep]
        dec = self.process(decisions)
        self.apply(dec)
        




