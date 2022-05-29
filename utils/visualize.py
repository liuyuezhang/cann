import cv2
import numpy as np
from utils.functions import l2_distance


def ax2bgr(ax):
    canvas = ax.figure.canvas
    canvas.draw()
    data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    return img


def data2img(data, size=300, color=True):
    img = process_img(data, smooth=False)
    h, w = img.shape
    # opencv: change (w, h) here
    img = cv2.resize(img, (size, int(size / w * h)))
    img = (img * 255).astype(np.uint8)
    if color:
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img


def process_img(img, smooth=True, eps=1e-16):
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + eps)
    if smooth:
        img = cv2.GaussianBlur(img, (3, 3), sigmaX=1, sigmaY=0)
    return img


def render_env(ax, qs, q_preds=None, gs=None):
    # qs
    qs = np.array(qs)
    ax.plot(qs[:, 0], qs[:, 1], 'b', label='pos')
    ax.plot(qs[-1, 0], qs[-1, 1], 'xb')

    # q_preds
    if q_preds is not None:
        q_preds = np.array(q_preds)
        ax.plot(q_preds[:, 0], q_preds[:, 1], 'r', label='pred')
        ax.plot(q_preds[-1, 0], q_preds[-1, 1], 'xr')

        error = l2_distance(q_preds[-1, :], qs[-1, :])
        ax.set_title('error={:.2f}'.format(error))

    # gs
    if gs is not None:
        gs = np.array(gs)[:, 0, 5]
        idx = (gs > 0)
        ax.plot(qs[idx, 0], qs[idx, 1], 'or')

    ax.legend(loc='lower right')
    res = ax2bgr(ax)
    return res
