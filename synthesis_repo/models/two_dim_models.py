import numpy as np
import matplotlib.pyplot as plt


class Robot():
    def __init__(self, time, x0, y0, tetta0, xf, yf, tettaf):
        self.x_history = [x0]
        self.y_history = [y0]
        self.tetta_history = [tetta0]
        self.time_history = [0.0]
        self.control_history = [(0.0, 0.0)]
        self.sim_time = time
        self.xf = xf
        self.yf = yf
        self.tettaf = tettaf
        self.eps = 0.01
        self.control_func = None
        self.dt = 0.01
        
    def set_control_function(self, f):
        def g(*args, **kwargs):
            return f(*args, **kwargs)
        self.control_func = g
    
    def set_dt(self, dt):
        self.dt = dt
    
    def simulate(self, dt=0.01):    
        t = dt
        su = 0
        x1, y1 = 5, 5 # obsticle with shape of circle with middle point at 5, 5
        rad = 2.5 # radius
        while t < self.sim_time:
            new_x, new_y, new_tetta = self.__euler_step(dt)
            if new_x == 1e+6 or new_y == 1e+6 or new_tetta == 1e+6: return 1e+6 # return right positive limit and exit
            try: 
                dr = math.pow(rad, 2) - math.pow(x1 - new_x, 2) - math.pow(y1 - new_y, 2)
                if dr > 0: su = su + 1
            except: pass
            estimation = self.estimate()
            if estimation > 1e+6: return 1e+6   
            self.x_history.append(new_x)
            self.y_history.append(new_y)
            self.tetta_history.append(new_tetta)
            self.time_history.append(t)
            if estimation < self.eps: return t
            t += dt
        return self.sim_time + estimation + su * dt
    
    def __euler_step(self, dt):
        x, y, tetta = self.__get_current_coords()
        dx, dy, dtetta = self.__get_right_parts(tetta)
        if dx > 1e+6 or dy > 1e+6 or dtetta > 1e+6: return 1e+6, 1e+6, 1e+6
        tilda_x = x + dt * dx
        tilda_y = y + dt * dy
        tilda_tetta = tetta + dt * dtetta
        
        tdx, tdy, tdtetta = self.__get_right_parts(tilda_tetta)
        x = x + (dx + tdx) * 0.5 * dt
        y = y + (dy + tdy) * 0.5 * dt
        tetta = tetta + (dtetta + tdtetta) * 0.5 * dt
        return x, y, tetta
    
    def __get_right_parts(self, tetta):
        current_coords = self.__get_current_coords()
        terminal_coords = self.__get_terminal_coords()
        state = terminal_coords - current_coords
        u1, u2 = self.control_func(state)
        self.clip_control(u1) # TODO: set control limits inside __init__
        self.clip_control(u2)
        self.control_history.append((u1, u2))
        right_x = (u1 + u2) * np.cos(tetta) * 0.5
        right_y = (u1 + u2) * np.sin(tetta) * 0.5
        right_tetta = (u1 - u2) * 0.5
        return right_x, right_y, right_tetta

    def clip_control(self, u):
        if u < -10: return -10
        elif u > 10: return 10
        else: return u
    
    def __get_current_coords(self,):
        return np.array([self.x_history[-1], self.y_history[-1], self.tetta_history[-1]])
    
    def __get_terminal_coords(self,):
        return np.array([self.xf, self.yf, self.tettaf])
    
    def estimate(self,):
        v0 = self.__get_current_coords()
        vf = self.__get_terminal_coords()
        return np.linalg.norm(vf - v0)
    
    def reset(self,):
        self.x_history = [self.x_history[0]]
        self.y_history = [self.y_history[0]]
        self.tetta_history = [self.tetta_history[0]]
        self.time_history = [0.0]
        self.control_histroy = [(0.0, 0.0)]
    
    def get_coords(self,):
        return (self.x_history, self.y_history)
    
    def get_control_in_time(self,):
        return (self.time_history, self.control_history)
    
    def plot_trajectory(self,):
        x, y = self.get_coords()
        fig = plt.figure()
        plt.plot(x, y, 'r')
        plt.xlabel('${x}$',fontsize=20)
        plt.ylabel('${y}$',fontsize=20)
        plt.legend(['${y}({x})$'],loc='upper right')
        plt.show()
        
    def plot_control_in_time(self,):
        t, u = self.get_control_in_time()
        fig = plt.figure()
        plt.plot(t, u, 'b')
        plt.xlabel('${t}$', fontsize=20)
        plt.ylabel('${u}$', fontsize=20)
        plt.legend(['${u}({t})$'],loc='upper right')
        plt.show()

