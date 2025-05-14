import dataclasses
import control
import numpy as np
import functools
from typing import Protocol, NamedTuple, Callable, Any

from numpy import ndarray


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
        ...


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
        q_matrices = [np.eye(a_matrices[0].shape[1]) for _ in range(len(a_matrices))]
    if r_matrices is None:
        r_matrices = [np.eye(1) for _ in range(len(a_matrices))]

    return [
        GainMatrixStruct(
            gain_location=x_1,
            gain_matrix=control.lqr(a, np.array([[0], [1]]), q, r)[0]
        )
        for x_1, a, q, r in zip(x_1_points, a_matrices, q_matrices, r_matrices)
    ]


def gain_scheduler(gain_matrix_structs: list[GainMatrixStruct], beta: float):
    def basis_function(x: float, x_i: float, gain: np.ndarray) -> np.ndarray:
        return np.exp(-beta * np.abs(x - x_i) ** 2) * gain

    funcs = [
        lambda x_: basis_function(x=x_, x_i=mat_struct.gain_location, gain=mat_struct.gain_matrix)
        for mat_struct in gain_matrix_structs
    ]

    def reduction(x_: np.ndarray):
        return functools.reduce(lambda f, g: f + g(x_), funcs[1:], funcs[0](x_))

    return reduction


if __name__ == "__main__":
    test_points = [np.pi, np.pi / 2, 0, -np.pi / 2]
    pendulum_ = Pendulum(l=1)
    for a in pendulum_jacobians(pendulum_, test_points):
        w, v = np.linalg.eig(a)
        print(a)
        print(w)

    for item in generate_optimal_gains(pendulum_, test_points):
        print(item.gain_matrix)
        print("\n")

    result = gain_scheduler(gain_matrix_structs=generate_optimal_gains(pendulum_, test_points), beta=1)
    for x_i_, a in zip(test_points, pendulum_jacobians(pendulum_, test_points)):
        print("==" * 10)
        print(f"{x_i_=} | {result(x_i_)=}")
        b = np.array([[0], [1]])
        print(f"Eigen Values -> {np.linalg.eig(a - b @ result(x_i_))[0]}")
        print("==" * 10)
        print("\n")
