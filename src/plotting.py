import dataclasses
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import numpy as np
from model import GaussianRbfGainScheduler, generate_optimal_gains, Pendulum

matplotlib.use("TkAgg")
plt.style.use("bmh")
plt.rcParams.update({"font.size": 7})

test_points = [np.pi, np.pi / 2, 0, -np.pi / 2]


@dataclasses.dataclass
class PendulumGaussianRbfPlotter:
    rbf_scheduler: GaussianRbfGainScheduler = GaussianRbfGainScheduler(
        beta=10,
        gain_matrix_structs=generate_optimal_gains(
            pendulum=Pendulum(l=1),
            x_1_points=test_points
        )
    )

    def plot_scheduling_gains_index(self, index: int) -> None:
        gain_struct = self.rbf_scheduler.gain_matrix_structs[index]
        optimal_gain_at_index = gain_struct.gain_matrix
        point = gain_struct.gain_location
        x_range = np.linspace(0, np.pi, 100)
        gains_series: np.ndarray = np.array(
            [self.rbf_scheduler.basis_function(x, point, optimal_gain_at_index) for x in x_range]
        )  # Shape  (n , 1, 2 )
        print(gains_series.shape)
        fig: Figure = plt.figure(figsize=(16 // 2, 5))

        ax_1: Axes = fig.add_subplot(211)
        ax_1.plot(x_range, gains_series[:, 0, 0], color="C4", linewidth=0.7, label="Gain 1")
        ax_1.axhline(y=optimal_gain_at_index[0, 0], linestyle="--", linewidth=0.7, color="C2", label="Optimal gain")
        ax_1.legend()
        ax_1.set_title(f"Gains At index {index}")

        ax_2: Axes = fig.add_subplot(212)
        ax_2.plot(x_range, gains_series[:, 0, 1], color="C4", linewidth=0.7, label="Gain 2")
        ax_2.axhline(y=optimal_gain_at_index[0, 1], linestyle="--", linewidth=0.7, color="C3", label="Optimal gain")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    plotter_1 = PendulumGaussianRbfPlotter()
    plotter_1.plot_scheduling_gains_index(1)
