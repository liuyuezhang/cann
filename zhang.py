import numpy as np
import cupy as cp
import utils.functions as f

import matplotlib.pyplot as plt
from utils.visualize import data2img, render_env
import cv2
from tqdm import tqdm


name = 'zhang'

# hyperparameters
alpha = 0.2  # dt / tau

ng = 64
Ng = ng * ng
A = 5.
lbd = 13.
abar = 1.05
beta = 3. / lbd**2

a = 1.0
b = 1.0

D, Dx, Dy = f.get_D(ng, l=0, e=None, require_Dx=True)
W = f.dog(D, A, abar * beta, A, beta)
Wx = f.dog_grad(D, Dx, A, abar * beta, A, beta)
Wy = f.dog_grad(D, Dy, A, abar * beta, A, beta)

idx = np.arange(Ng * Ng)
np.random.shuffle(idx)
idx1 = idx[0:int(Ng * Ng / 2)]
idx2 = idx[int(Ng * Ng / 2):]

Wx = Wx.reshape(-1)
Wx[idx1] = 0.
Wx = Wx.reshape(Ng, Ng)

Wy = Wy.reshape(-1)
Wy[idx2] = 0.
Wy = Wy.reshape(Ng, Ng)

W, Wx, Wy = cp.asarray(W), cp.asarray(Wx), cp.asarray(Wy)


B = 1
joystick = False
if joystick is False:
    from utils.agent import RandomAgent
    agent = RandomAgent(batch_size=B, mu=0.2, sigma=3.)
else:
    from utils.agent import JoyStickAgent
    agent = JoyStickAgent(batch_size=B)

from utils.env import VecArenaEnv2D
env = VecArenaEnv2D(width=2.2, height=2.2, batch_size=B, cue='none', dt=0.02)

u = cp.random.normal(scale=1. / np.sqrt(Ng), size=(B, Ng)).astype(cp.float32)
qs = []
gs = []

q = env.reset()

# run model (init 800 steps)
for t in tqdm(range(500 + 4000)):
    # action
    if joystick:
        cmd = cv2.waitKey(2) & 0xFF
        if cmd == ord('q'):
            break
        elif cmd == ord('r'):
            qs = []
            gs = []
        else:
            action = agent.act(cmd)
            cv2.waitKey(1)
    else:
        action = agent.act()

    # step
    q, ob = env.step(action)
    vel, theta = ob[0], ob[1]
    vx, vy = vel * np.cos(theta), vel * np.sin(theta)
    v = np.stack((vx, vy), axis=-1)  # shape: (B, 2)

    vx = cp.asarray(np.expand_dims(vx, -1))
    vy = cp.asarray(np.expand_dims(vy, -1))

    # input
    h = a * ((f.relu(vx) + cp.sign(f.relu(vx)) * u) @ Wx + (f.relu(-vx) + cp.sign(f.relu(-vx)) * u) @ (-Wx)
             + (f.relu(vy) + cp.sign(f.relu(vx)) * u) @ Wy + (f.relu(vy) + cp.sign(f.relu(-vy)) * u) @ (-Wy))
    # rnn
    r = f.relu(u @ W + b + h)
    u = (1. - alpha) * u + alpha * r

    if t >= 500:
        # log
        gs.append(cp.asnumpy(r))
        qs.append(q)  # no init q

        # plots
        if t % 10 == 1:
            # visualize
            neural_img = data2img(cp.asnumpy(r[0, :]).reshape(ng, ng), size=256)
            # W_img = data2img(cp.asnumpy(WR), size=256)
            cv2.imshow("neural space", neural_img)
            # cv2.imshow("W", W_img)
            cv2.waitKey(1)

            # display env
            fig, ax = env.plot()
            img_env = render_env(ax, np.array(qs)[:, 0, :], q_preds=None, gs=gs)
            plt.close(fig)
            cv2.imshow('env', img_env)

            # # display single neuron
            # img_single = render_single(t=t+1, ys=np.array(gs)[:, 0, 132])
            # cv2.imshow('single', img_single)


# gs = np.array(gs).swapaxes(0, 1)  # shape: (B, T, Ng)
# qs = np.array(qs).swapaxes(0, 1)  # shape: (B, T, 2)
#
# gs = gs[:, :, 0:32]
# print(gs.shape, qs.shape)
#
# # compute ratemap
# activations = calc_ratemap(gs, qs, widths=(2.2, 2.2))
# # calc fig
# rm_fig = plot_ratemap(activations, n_plots=len(activations))
# plt.savefig("grid.png")
# # cv2.imshow("grid", rm_fig)
# # plt.imshow(cp.asnumpy(gs)[0, -1, :].reshape(ng, ng))
# # plt.colorbar()
