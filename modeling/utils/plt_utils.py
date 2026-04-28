import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from utils.exp_utils import load_hp
import matplotlib.pyplot as plt

def standard_2d_ax(w=4, h=4):
    # Create figure and 3D axes
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig, ax

def no_ticks_2d_ax():
    fig, ax = standard_2d_ax()
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax

def empty_2d_ax():
    fig, ax = standard_2d_ax()
    ax.set_axis_off()  # hides axes, ticks, labels, etc.
    return fig, ax

def ax_3d_no_grid():
    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    return fig, ax

def empty_3d():
    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()  # hides axes, ticks, labels, etc.
    return fig, ax