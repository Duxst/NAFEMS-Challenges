import numpy as np
from numpy.random import uniform
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

num_of_frames = 50
num_of_samples = 150_000

x = uniform(low=-1.0, high=1.0, size=num_of_samples)
y = uniform(low=-1.0, high=1.0, size=num_of_samples)

radius = 1.0
theta = np.linspace(0, 2 * np.pi, 1000)
x_circle = radius * np.cos(theta)
y_circle = radius * np.sin(theta)

fig, ax = plt.subplots(figsize=(4.8, 4.8))

def animate(i):
    print(i)
    ax.clear()
    x_samples = x[:int(num_of_samples*(i+1)/num_of_frames)]
    y_samples = y[:int(num_of_samples * (i+1) / num_of_frames)]
    inside_circle = np.sqrt(x_samples ** 2 + y_samples ** 2) <= 1
    x_inside = x_samples[inside_circle]
    x_outside = x_samples[~inside_circle]
    y_inside = y_samples[inside_circle]
    y_outside = y_samples[~inside_circle]
    ax.scatter(x_inside, y_inside, 1, c='b', alpha=0.5)
    ax.scatter(x_outside, y_outside, 1, c='r', alpha=0.5)
    ax.plot(x_circle, y_circle, 'k')
    ax.axis('equal')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title(fr'$n = ${len(x_samples):,}    $\pi \approx {4*len(x_inside)/len(x_samples):.4f}$')
    return


ani = FuncAnimation(fig, animate, interval=200, repeat=True, frames=num_of_frames)
ani.save("TLI.gif", dpi=300, writer=PillowWriter(fps=5))