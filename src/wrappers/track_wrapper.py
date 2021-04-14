from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener, shape)
import Box2D
from pyglet import gl
import gym
import numpy as np
import math
import pyglet
from gym.envs.classic_control import rendering

pyglet.options["debug_gl"] = False


class TrackWrapper(gym.Wrapper):
    def __init__(self, env, seed):
        super().__init__(env)
        env.seed = lambda: env.seed(seed)
        self.env = env


STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 1  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
FPS = 50  # Frames per second
ZOOM = 1.5  # Camera zoom
ZOOM_FOLLOW = False  # Set to False for fixed view (don't use zoom)


def render_mini(self, mode="human"):
    assert mode in ["human", "state_pixels", "rgb_array"]
    if self.viewer is None:

        self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
        self.score_label = pyglet.text.Label(
            "0000",
            font_size=36,
            x=-100,
            y=-100,
            anchor_x="left",
            anchor_y="center",
            color=(255, 255, 255, 255),
        )
        self.transform = rendering.Transform()

    if "t" not in self.__dict__:
        return  # reset() not called yet

    # Animate zoom first second:
    zoom = ZOOM
    # zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
    scroll_x = self.car.hull.position[0]
    scroll_y = self.car.hull.position[1]

    angle = -self.car.hull.angle
    vel = self.car.hull.linearVelocity
    if np.linalg.norm(vel) > 0.5:
        angle = math.atan2(vel[0], vel[1])
    self.transform.set_scale(zoom, zoom)
    self.transform.set_translation(
        WINDOW_W / 2,
        WINDOW_H / 2
    )
    # self.transform.set_rotation(angle)

    self.car.draw(self.viewer, mode != "state_pixels")
    translate_indicator = rendering.Transform(translation=self.car.hull.position)
    self.viewer.draw_circle(75, 75, True, color=(0, 0, 0.8)).add_attr(translate_indicator)

    arr = None
    win = self.viewer.window
    win.switch_to()
    win.dispatch_events()

    win.clear()
    t = self.transform
    if mode == "rgb_array":
        VP_W = VIDEO_W
        VP_H = VIDEO_H
    elif mode == "state_pixels":
        VP_W = STATE_W
        VP_H = STATE_H
    else:
        pixel_scale = 1
        if hasattr(win.context, "_nscontext"):
            pixel_scale = (
                win.context._nscontext.view().backingScaleFactor()
            )  # pylint: disable=protected-access
        VP_W = int(pixel_scale * WINDOW_W)
        VP_H = int(pixel_scale * WINDOW_H)

    gl.glViewport(0, 0, VP_W, VP_H)
    t.enable()
    self.render_road()
    for geom in self.viewer.onetime_geoms:
        geom.render()
    self.viewer.onetime_geoms = []
    t.disable()
    self.render_indicators(WINDOW_W, WINDOW_H)

    if mode == "human":
        win.flip()
        return self.viewer.isopen

    image_data = (
        pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
    )
    arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep="")
    arr = arr.reshape(VP_H, VP_W, 4)
    arr = arr[::-1, :, 0:3]

    return arr


def syncStep(envs, action):
    for e in envs:
        e.step(action)


def syncReset(envs):
    for e in envs:
        e.reset()


def seed_uniform(a, b, seed):
    np.random.seed(seed)
    return np.random.randint(a*100, b*100)/100


class PseudoRand():
    def __init__(self):
        self.uniform = None


class TrackMini(gym.Wrapper):
    def __init__(self, create_env, seed):
        env = create_env()
        mini = create_env()
        super().__init__(env)
        env.np_random = PseudoRand()
        mini.np_random = PseudoRand()
        env.np_random.uniform = lambda a, b: seed_uniform(a, b, seed)
        mini.np_random.uniform = lambda a, b: seed_uniform(a, b, seed)
        mini.render = lambda mode="human": render_mini(mini, mode)
        self.env = env
        self.mini = mini

    def reset(self):
        self.env.reset()
        self.mini.reset()

    def step(self, action):
        self.mini.step(action)
        return self.env.step(action)
