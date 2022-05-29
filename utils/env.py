import gym
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString
from geopandas import GeoSeries


class VecArenaEnv2D(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, width=2.2, height=2.2, batch_size=200, random_init=False, init_width=None, init_height=None, cue='none', dt=0.02):
        super(VecArenaEnv2D, self).__init__()
        # objects
        self.arena = Polygon([(width/2, -height/2), (width/2, height/2), (-width/2, height/2), (-width/2, -height/2)])
        self.obstacles = []

        # coordinates
        self.width = width
        self.height = height
        self.random_init = random_init
        self.init_width = init_width if init_width is not None else self.width
        self.init_height = init_height if init_height is not None else self.height

        # parameters
        self.batch_size = batch_size
        self.cue = cue
        self.range = np.sqrt(self.width**2 + self.height**2)  # reachable range of the cue
        self.border_region = 0.03  # meters
        self.dt = dt  # time step increment (seconds)

        # agent
        self.q = np.zeros((self.batch_size, 2), dtype=np.float32)
        self.length = np.ones(self.batch_size, dtype=np.float32) * np.nan

        # state variables
        self.v = None
        self.theta = None

    def get_cue(self):
        # calc mouse and light, have to calc the dist even for none cue since we need to turn around corner
        mouse = GeoSeries([Point(self.q[i]) for i in range(self.batch_size)])
        end_coord = self.q + self.range * np.stack([np.cos(self.theta), np.sin(self.theta)], axis=-1)
        light = GeoSeries(LineString([self.q[i], end_coord[i]]) for i in range(self.batch_size))
        # calc end points
        min_dist = np.ones(self.batch_size, dtype=np.float32) * self.range
        for obj in [self.arena] + self.obstacles:
            p = light.intersection(obj.boundary)
            dist = mouse.distance(p)
            idx = (dist < min_dist).tolist()
            min_dist[idx] = dist[idx]
        self.length = min_dist  # for visualization
        if self.cue == 'none':
            return None
        if self.cue == 'dist':
            return min_dist
        else:
            raise NotImplementedError

    def add_obstacles(self, obstacles):
        self.obstacles += obstacles

    def reset(self, q0=None):
        if q0 is not None:
            self.q = q0.astype(np.float32)
        else:
            if self.random_init:
                self.q[:, 0] = np.random.uniform(-self.init_width / 2, self.init_width / 2, self.batch_size)
                self.q[:, 1] = np.random.uniform(-self.init_width / 2, self.init_width / 2, self.batch_size)
            else:
                self.q[:, 0] = np.zeros(self.batch_size, dtype=np.float32)
                self.q[:, 1] = np.zeros(self.batch_size, dtype=np.float32)
            valid = self.is_valid(GeoSeries([Point(p) for p in self.q]))
            while (~valid).sum() > 0.5:
                self.q[~valid, 0] = np.random.uniform(-self.init_width / 2, self.init_width / 2, (~valid).sum())
                self.q[~valid, 1] = np.random.uniform(-self.init_width / 2, self.init_width / 2, (~valid).sum())
                valid = self.is_valid(GeoSeries([Point(p) for p in self.q]))

        self.v = np.zeros(self.batch_size, dtype=np.float32)
        # todo: changed here
        # self.theta = np.random.uniform(0, 2 * np.pi, self.batch_size).astype(np.float32)
        self.theta = np.zeros(self.batch_size, dtype=np.float32)
        self.get_cue()
        # return s_0
        return self.q.copy()

    def step(self, action):
        # action
        v, omega = action

        # update theta
        dtheta = self.dt * omega
        near_wall = self.length < self.border_region
        dtheta[near_wall] += np.pi / 5
        self.theta = np.mod(self.theta + dtheta, 2 * np.pi)

        # step
        dq = self.dt * v[:, None] * np.stack([np.cos(self.theta), np.sin(self.theta)], axis=-1)
        next_q = self.q + dq

        # update
        valid = self.is_valid(GeoSeries([Point(p) for p in next_q]))
        self.v[valid] = v[valid]
        self.v[~valid] = 0.
        self.q[valid] = next_q[valid]

        # return s_{t+1}, actual ob_t
        ob = self.v.copy(), self.theta.copy(), self.get_cue()
        return self.q.copy(), ob

    def is_valid(self, points):
        valid = np.array([True] * len(points), dtype=np.bool)
        for obstacle in self.obstacles:
            valid &= (~points.within(obstacle))
        valid &= points.within(self.arena)
        return np.array(valid.tolist(), dtype=np.bool)

    def render(self, mode='human', close=False):
        pass

    def plot(self, i=0, msg="env", us=None):
        fig, ax = plt.subplots(figsize=(5, 5))
        # plot arena
        xs, ys = self.arena.exterior.xy
        ax.plot(xs, ys)
        # plot obstacles
        for obstacle in self.obstacles:
            xs, ys = obstacle.exterior.xy
            ax.fill(xs, ys, alpha=0.75)
        # plot place cell
        if us is not None:
            ax.scatter(us[:, 0], us[:, 1], s=20, alpha=1, c='darkgrey')
        # plot agent
        ax.scatter(self.q[i, 0], self.q[i, 1])
        # plot light
        if self.length[i] is not np.nan:
            end_coord = self.q[i] + self.length[i] * np.stack([np.cos(self.theta), np.sin(self.theta)], axis=-1)
            light = LineString([self.q[i], end_coord[i]])
            xs, ys = light.xy
            ax.plot(xs, ys)
        # title
        ax.set_title(msg)
        # res = ax2bgr(ax)
        # plt.close(fig)
        return fig, ax
