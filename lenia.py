try:
    import cupy as np
    import cupyx.scipy.signal as signal

    print("CuPy loaded successfully. Running in GPU mode.")


    def pure_array(arr):
        return arr.get()
except Exception as ex:
    print("CuPy failed to load. Running in CPU mode.")
    import numpy as np
    import scipy.signal as signal


    def pure_array(arr):
        return arr

import matplotlib.pyplot as plt
import matplotlib.animation as anim
from time import monotonic


def characteristics_plot(kernel, growth=None, clipping=None):
    fig, ax = plt.subplots(1, 3)
    # kernel
    ax[0].set_title("Kernel")
    ax[0].imshow(pure_array(kernel), vmin=0)

    # growth function plot
    ax[1].set_title("Growth function")
    x_growth = np.linspace(0, 1)
    y_growth = growth(x_growth)
    ax[1].plot(pure_array(x_growth), pure_array(y_growth))

    # clipping function plot
    ax[2].set_title("Clipping function")
    x_clip = np.linspace(-10, 10)
    y_clip = clipping(x_clip)
    ax[2].plot(pure_array(x_clip), pure_array(y_clip))

    fig.tight_layout()
    plt.show(block=False)


class Lenia:
    def __init__(self,
                 kernel_radius=10, kernel_func=None,
                 world=None, random_world_size=64,
                 update_freq=10, growth_func=None,
                 clipping_func=None
                 ):

        if world is None:
            np.random.seed(int(monotonic()))
            self.__world__ = np.random.rand(random_world_size, random_world_size)
        else:
            self.__world__ = world

        if kernel_func is None:
            self.__kernel_func__ = lambda x: np.exp(-((x - 0.5) / 0.15) ** 2 / 2)
        else:
            self.__kernel_func__ = kernel_func

        if growth_func is None:
            self.__growth_func__ = lambda x: (np.exp(-((x - 0.135) / 0.035) ** 2 / 2)) * 2 - 1
        else:
            self.__growth_func__ = growth_func

        if clipping_func is None:
            self.__clipping_func__ = lambda x: np.clip(x, 0, 1)
        else:
            self.__clipping_func__ = clipping_func

        # calculate kernel
        R = kernel_radius
        x_arr, y_arr = np.mgrid[-R:R, -R:R] + 1
        D = np.sqrt(x_arr ** 2 + y_arr ** 2) / R
        self.__kernel__ = self.__kernel_func__(D)
        self.__kernel__ = self.__kernel__ / np.sum(self.__kernel__)

        # plot characteristic functions
        characteristics_plot(self.__kernel__, self.__growth_func__, self.__clipping_func__)

        # calculate time step
        self.__dt__ = 1 / update_freq

        # plotting init
        self.__frames__ = []
        self.fig = plt.figure()
        self.plot = plt.imshow(pure_array(self.__world__), vmin=0)

    def __growth_update__(self, U):
        return self.__growth_func__(U)

    def __update_world__(self):
        self.__frames__.append(self.__world__)  # save last state
        U = signal.convolve2d(self.__world__, self.__kernel__, mode="same", boundary="wrap")
        self.__world__ = self.__world__ + self.__dt__ * self.__growth_update__(U)
        self.__world__ = self.__clipping_func__(self.__world__)

    def calculate_frames(self, num_of_frames):
        t0 = monotonic()
        for i in range(num_of_frames):
            self.__update_world__()
        time_taken = monotonic() - t0
        print(f"Calculations done in {time_taken} ms. Time per frame: {time_taken / num_of_frames} ms.")

    def __update_plot__(self, step):  # step required by func anim in matplotlib
        if not self.__frames__:
            raise ValueError("No frames calculated. Run .calculate_frames first!")
        self.plot.set_array(pure_array(self.__frames__[step]))
        return self.plot,

    def animate(self):
        print("Starting animation generation. This might take some time.")
        return anim.FuncAnimation(self.fig, self.__update_plot__, frames=len(self.__frames__), interval=1)


def world_gen(size, R):
    w = np.zeros((size, size))
    x, y = np.mgrid[0:R, 0:R] + (size - R) // 2
    w[x, y] = np.random.rand(*w[x, y].shape)
    return w


def main():
    l = Lenia(
        world=world_gen(100, 10)
    )
    l.calculate_frames(1000)
    a = l.animate()
    a.save(f"runs/lenia_run_{int(monotonic())}.mp4", writer="ffmpeg", fps=24)


if __name__ == "__main__":
    main()
