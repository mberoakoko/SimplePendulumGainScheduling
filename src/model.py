import dataclasses
import control
import numpy as np
import functools
from typing import Protocol, NamedTuple, Callable, Any


class System(Protocol):
    def update(self, t: float, x: np.ndarray, u: np.ndarray, params: dict) -> np.ndarray:
        ...

    def output(self, t: float, x: np.ndarray, u: np.ndarray, params: dict) -> np.ndarray:
        ...


@dataclasses.dataclass(frozen=True)
class Pendulum(System):
    l: int

    def update(self, t: float, x: np.ndarray, u: np.ndarray, params: dict) -> np.ndarray:
        x_1, x_2 = x
        return np.array([
            x_2,
            -9.81 / self.l * np.sin(x_1) + u
        ])

    def output(self, t: float, x: np.ndarray, u: np.ndarray, params: dict) -> np.ndarray:
        return x

    def generate_system(self) -> control.NonlinearIOSystem:
        return control.NonlinearIOSystem(
            self.update, self.output,
            inputs=("u",), outputs=("x",),
            states=("theta", "theta_dot"),
            name="Pendulum System"
        )


def pendulum_jacobians(pendulum: Pendulum, x_1_points: list[float]) -> list[np.ndarray]:
    def jacobian(x: float):
        return np.array([
            [0, 1],
            [-(9.81 / pendulum.l) * np.cos(x), 0]
        ])

    return list(map(jacobian, x_1_points))


class GainMatrixStruct(NamedTuple):
    gain_location: float
    gain_matrix: np.ndarray


def generate_optimal_gains(
        pendulum: Pendulum,
        x_1_points: list[float],
        q_matrices: list[np.ndarray] | None = None,
        r_matrices: list[np.ndarray] | None = None) -> list[GainMatrixStruct]:
    a_matrices = pendulum_jacobians(pendulum, x_1_points)
    if q_matrices is None:
        q_matrices = [np.diag((10, 1000)) for _ in range(len(a_matrices))]
    if r_matrices is None:
        r_matrices = [10 * np.eye(1) for _ in range(len(a_matrices))]

    return [
        GainMatrixStruct(
            gain_location=x_1,
            gain_matrix=control.lqr(a, np.array([[0], [1]]), q, r)[0]
        )
        for x_1, a, q, r in zip(x_1_points, a_matrices, q_matrices, r_matrices)
    ]


def gain_scheduler(gain_matrix_structs: list[GainMatrixStruct], beta: float):
    def basis_function(x: float | np.ndarray, x_i: float, gain: np.ndarray) -> np.ndarray:
        return np.exp(-beta * np.abs(x - x_i) ** 2) * gain

    funcs = [
        lambda x_: basis_function(x=x_, x_i=mat_struct.gain_location, gain=mat_struct.gain_matrix)
        for mat_struct in gain_matrix_structs
    ]

    def reduction(x_: np.ndarray):
        return functools.reduce(lambda f, g: f + g(x_), funcs[1:], funcs[0](x_))

    return reduction


@dataclasses.dataclass
class GaussianRbfGainScheduler:
    beta: float
    gain_matrix_structs: list[GainMatrixStruct]

    def basis_function(self, x: float, x_i: float, gain_matrix: np.ndarray) -> np.ndarray:
        return np.exp(-self.beta * np.abs(x - x_i) ** 2) * gain_matrix

    def gain(self, x: np.ndarray | float):
        locs = (mat_struct.gain_location for mat_struct in self.gain_matrix_structs)
        gains = (mat_struct.gain_matrix for mat_struct in self.gain_matrix_structs)
        total_gains = np.array([self.basis_function(x, location, gain) for location, gain in zip(locs, gains)])
        return np.sum(total_gains, axis=0)

    def __repr__(self):
        repr_str = f"{self.__class__.__name__}\n"
        for gain_mat in self.gain_matrix_structs:
            repr_str += f"Gain Loc => {gain_mat.gain_location}\n"
            repr_str += f"Gain Mat => {gain_mat.gain_matrix.__str__()}\n"
        return repr_str


@dataclasses.dataclass
class PendulumController(System):
    gain_scheduler: GaussianRbfGainScheduler

    def update(self, t: float, x: np.ndarray, u: np.ndarray, params: dict) -> np.ndarray:
        ...

    def output(self, t: float, x: np.ndarray, u: np.ndarray, params: dict) -> np.ndarray:
        return -self.gain_scheduler.gain(x) @ x

    def generate_system(self) -> control.NonlinearIOSystem:
        return control.NonlinearIOSystem(
            None, self.output,
            unputs=("", ), outputs=("", ),
            states=("theta", "theta_dot"),
            name="Pendulum Controller"
        )


@dataclasses.dataclass
class ClosedLoopPendulum(System):
    pendulum_system: control.NonlinearIOSystem
    pendulum_controller: control.NonlinearIOSystem

    def generate_system(self) -> control.NonlinearIOSystem:
        return control.interconnect(
            syslist=[self.pendulum_system, self.pendulum_controller],
            inplist=("x", ""), outlist=("")
        )


if __name__ == "__main__":
    test_points = [np.pi, np.pi / 2, 0, -np.pi / 2]
    pendulum_ = Pendulum(l=10)

    for item in generate_optimal_gains(pendulum_, test_points):
        print(item.gain_matrix)
        print("\n")

    rbf_gain_scheduler = GaussianRbfGainScheduler(
        beta=1,
        gain_matrix_structs=generate_optimal_gains(pendulum_, test_points)
    )

    print(rbf_gain_scheduler)
    print(f"{rbf_gain_scheduler.gain(np.pi)}=")
    b = np.array([[0], [-1]])
    x_sweep = np.linspace(-np.pi, np.pi, 3)
    for a, x_test in zip(pendulum_jacobians(pendulum_, test_points), test_points):
        w, v = np.linalg.eig(a)
        w_1, _ = np.linalg.eig(a - b @ rbf_gain_scheduler.gain(x_test))
        print(f"Eigen values {w}")
        print(f"Closed loop values {w_1}")
