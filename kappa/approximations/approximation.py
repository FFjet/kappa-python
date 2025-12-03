from __future__ import annotations

import math
from typing import Iterable, Callable

import numpy as np
from scipy import special, integrate

from .. import constants
from ..interactions import Interaction
from ..models import (
    ModelsOmega,
    ModelsKExch,
    ModelsKVT,
    ModelsKVV,
    ModelsKDiss,
    ModelsProbVV,
    ModelsProbVT,
    ModelsProbDiss,
    ModelsCsElastic,
    ModelsCsVT,
    ModelsCsDiss,
)
from ..exceptions import DataNotFoundException, ModelParameterException
from ..numerics import integrate_semi_inf as integrate_semi_inf_cpp
from ..particles import Molecule, Particle


class Approximation:
    """Python port of the approximation routines using NumPy/SciPy."""

    # Precompute a small factorial table to mirror the Armadillo fixed-size vector
    factorial_table: np.ndarray = np.array([math.factorial(i) for i in range(70)], dtype=float)
    cc_ar: np.ndarray = np.array([0.0, 1.0, 1.5, 1.83333333, 2.08333333333333], dtype=float)
    cc_cl: np.ndarray = np.array([0.5, 1.0, 1.1666666666666667, 1.333333333333333, 1.43333333333333], dtype=float)
    p4e_n2n_diss_bjk: np.ndarray = np.array(
        [
            [-4.102280e01, -1.132030e05, -5.081880e05, 6.584820e07, 2.009300e00],
            [-3.952490e-01, 3.502120e03, 1.616070e04, -3.360490e06, 6.080030e-02],
            [2.810930e-02, -3.104770e01, 2.812760e01, 5.289150e04, -3.362940e-03],
            [-2.727510e-04, 6.024030e-02, -2.413860e00, -2.574970e02, 3.166530e-05],
        ]
    )
    p4e_o2o_diss_bjk: np.ndarray = np.array(
        [
            [-27.0145, -3.19019, 1.05845, -0.126656, 0.00721311, -0.000211765, 3.10206e-06, -1.79668e-08],
            [-59620.0, 2178.93, -11.2525, 1.59843, -0.177455, 0.00687121, -0.000119773, 7.84053e-07],
            [0.675455, 0.376682, -0.121598, 0.0144864, -0.00082352, 2.41602e-05, -3.53967e-07, 2.05237e-09],
        ]
    )
    p4e_n2n_vt_bij1: np.ndarray = np.array(
        [
            [-3.189200e01, 4.915400e-01, 3.303700e-03, -9.362800e-05, -3.517000e00],
            [-1.413600e04, 4.269700e02, -1.129500e01, 8.816800e-02, 2.445900e03],
            [7.545700e03, -2.282000e03, 7.615200e01, -8.509700e-01, 6.103300e03],
            [-4.588900e05, 1.564300e05, -4.100500e03, 3.705600e01, -4.393800e05],
            [9.796900e-01, -5.611100e-02, -3.362100e-04, 1.052400e-05, 3.810500e-01],
        ]
    )
    p4e_n2n_vt_bij2: np.ndarray = np.array(
        [
            [-2.620600e01, 8.255100e-01, 6.123000e-04, -9.805500e-05, -7.161100e00],
            [-2.265800e03, 1.615800e03, -3.114100e01, 2.170400e-01, -7.257200e03],
            [2.338600e05, 1.571600e04, -1.909100e02, 6.338200e-01, -1.583300e05],
            [-6.228100e06, -3.163500e05, 2.992200e03, -2.897700e00, 3.819000e06],
            [3.333100e-01, -9.797200e-02, 3.756200e-06, 1.106000e-05, 8.232000e-01],
        ]
    )
    p4e_n2n_vt_cij2: np.ndarray = np.array(
        [
            [-5.611300e00, -4.912300e-01, 6.910000e-03, -3.185400e-05, 4.307800e00],
            [-7.957600e03, -8.055500e02, 1.344900e01, -8.712100e-02, 6.528700e03],
            [-1.589600e05, -1.449600e04, 2.410800e02, -1.532100e00, 1.225500e05],
            [3.886100e06, 3.431100e05, -5.689700e03, 3.633400e01, -2.949500e06],
            [6.078700e-01, 6.013200e-02, -8.229600e-04, 3.493500e-06, -5.128000e-01],
        ]
    )
    p4e_n2n_vt_bij3: np.ndarray = np.array(
        [
            [-3.371800e02, -2.602000e01, 4.249100e-01, -2.700200e-03, 2.283700e02],
            [-2.859800e04, -2.867000e02, -2.359700e00, 4.173600e-02, 1.094600e04],
            [-2.011800e06, -2.042300e05, 3.468900e03, -2.291500e01, 1.653800e06],
            [3.856700e07, 4.253000e06, -7.432600e04, 5.018800e02, -3.309400e07],
            [3.598700e01, 2.942500e00, -4.767100e-02, 3.007900e-04, -2.605400e01],
        ]
    )
    p4e_n2n_vt_cij3: np.ndarray = np.array(
        [
            [3.525100e01, 3.070400e00, -4.988000e-02, 3.179500e-04, -2.673400e01],
            [6.733300e02, -3.869300e01, 1.178400e00, -1.024700e-02, -9.191300e01],
            [2.211900e05, 2.316200e04, -3.953300e02, 2.627600e00, -1.849700e05],
            [-4.030600e06, -4.639000e05, 8.147400e03, -5.540000e01, 3.549500e06],
            [-4.193900e00, -3.546000e-01, 5.729100e-03, -3.640900e-05, 3.126500e00],
        ]
    )
    p4e_n2n_vt_bij4: np.ndarray = np.array(
        [
            [-5.144800e02, -1.811900e01, 2.071900e-01, -9.692600e-04, 2.553900e02],
            [3.950300e04, 2.329000e03, 2.329000e03, 2.329000e03, -2.567600e04],
            [-5.504500e06, -2.607000e05, 3.365900e03, -1.814000e01, 3.174700e06],
            [1.380000e08, 6.692000e06, -8.798600e04, 4.828400e02, -8.019000e07],
            [4.876200e01, 1.742200e00, -1.924000e-02, 8.480100e-05, -2.511100e01],
        ]
    )
    p4e_n2n_vt_cij4: np.ndarray = np.array(
        [
            [3.580700e01, 1.501200e00, -1.845200e-02, 9.490200e-05, -1.966800e01],
            [-4.267800e03, -2.086000e02, 2.816000e00, -1.594900e-02, 2.471900e03],
            [3.556600e05, 1.801800e04, -2.461400e02, 1.411600e00, -2.094400e05],
            [-3.901100e00, -1.633400e-01, 5.732900e03, -3.354800e01, 4.690900e06],
            [4.876200e01, 1.742200e00, 2.005500e-03, -1.027200e-05, 2.139000e00],
        ]
    )
    p4e_n2n_vt_bij5: np.ndarray = np.array(
        [
            [-1.913900e03, -5.248100e01, 4.885700e-01, -1.879900e-03, 9.005900e02],
            [5.118300e04, 1.408200e03, -9.481500e00, 3.505700e-03, -2.518000e04],
            [2.867900e07, 8.497200e05, -8.581400e03, 3.896300e01, -1.395200e07],
            [-1.343500e09, -4.098400e07, 4.169100e05, -1.873600e03, 6.606300e08],
            [2.166000e02, 6.183200e00, -5.853200e-02, 2.295400e-04, -1.041800e02],
        ]
    )
    p4e_n2n_vt_cij5: np.ndarray = np.array(
        [
            [1.118000e02, 3.463900e00, -3.530000e-02, 1.535600e-04, -5.523700e01],
            [-7.693000e03, -2.617800e02, 2.849600e00, -1.316300e-02, 3.912200e03],
            [-9.314000e05, -2.472600e04, 2.224700e02, -8.793600e-01, 4.399400e05],
            [-2.386600e07, 1.438500e06, -1.402100e04, 5.974700e01, -2.386600e07],
            [2.166000e02, -2.386600e07, 4.526400e-03, -2.007900e-05, 6.831900e00],
        ]
    )
    p4e_n2n_vt_bij6: np.ndarray = np.array(
        [
            [7.730100e01, 7.730100e01, 7.730100e01],
            [1.449000e03, 1.449000e03, 4.050100e-01],
            [2.492500e08, -9.691600e06, 9.060600e04],
            [-6.695300e02, 2.100000e01, -1.808600e-01],
        ]
    )
    p4e_n2n_vt_cij6: np.ndarray = np.array(
        [
            [-2.929200e00, 9.561900e-02, -8.020000e-04],
            [-1.393200e02, 4.419500e00, -3.464200e-02],
            [-5.928900e06, 2.393900e05, -2.354800e03],
            [1.660300e01, -5.307500e-01, 4.228100e-03],
        ]
    )
    p4e_o2o_vt_Cijk1: np.ndarray = np.array(
        [
            [
                [-26.18993227, 0, 0, 0],
                [-1.69828917, 0, 0, 0],
                [3.349076e19, 0, 0, 0],
                [-3.946126e20, 0, 0, 0],
                [1.391056e19, 0, 0, 0],
            ],
            [
                [7.833311e00, 0, 0, 0],
                [3.712214e00, 0, 0, 0],
                [3.573261e20, 0, 0, 0],
                [6.433503e20, 0, 0, 0],
                [-2.901352e19, 0, 0, 0],
            ],
            [
                [3.716395e-01, 0, 0, 0],
                [1.058709e-01, 0, 0, 0],
                [-5.312491e19, 0, 0, 0],
                [3.754092e19, 0, 0, 0],
                [-1.189832e18, 0, 0, 0],
            ],
        ]
    )
    p4e_o2o_vt_Cijk2: np.ndarray = np.array(
        [
            [
                [-1.945676e01, -3.380076e00, 8.985159e01, 5.853646e-02],
                [-1.487120e00, -5.401834e-01, -5.333457e01, -9.543789e-02],
                [1.505136e21, 1.621622e21, -1.066183e21, 2.169160e20],
                [-1.532916e20, -4.105380e19, 1.185042e21, -1.748646e18],
                [-4.838473e18, 9.529399e18, 5.290114e19, 6.021495e16],
            ],
            [
                [1.447822e01, -4.332225e01, -3.481680e02, -1.641860e-01],
                [3.266821e01, -1.846219e-01, 1.190313e02, 2.225885e-01],
                [1.522507e21, 2.654567e21, -3.528669e21, -2.861293e20],
                [-1.533872e21, -1.522587e20, -3.124046e21, 2.322209e19],
                [-3.762650e19, 1.955942e19, 8.847719e18, -8.252347e17],
            ],
            [
                [8.865765e-01, -1.142588e00, -4.530804e00, 4.732032e-03],
                [-1.856692e-01, -1.925642e-01, 3.958191e00, 1.353007e-02],
                [-2.027215e21, 2.381051e21, -1.248596e20, -4.014395e19],
                [1.819921e19, 3.708631e19, 2.031805e19, -6.021031e17],
                [1.530423e18, -1.859900e18, -8.207403e18, 9.708459e15],
            ],
        ]
    )
    p4e_o2o_vt_Cijk3: np.ndarray = np.array(
        [
            [
                [-3.993663e00, 2.684515e00, -1.009927e05, -2.836760e-01],
                [-3.030965e00, -4.594443e00, 3.590903e04, 7.104764e-02],
                [5.492061e21, 1.212196e21, 9.092488e21, 1.540038e20],
                [1.308503e20, 1.831856e20, 1.079540e22, -5.608629e18],
                [2.160753e19, -1.465914e19, 5.483520e21, 1.142128e17],
            ],
            [
                [-7.575157e01, -9.234026e00, 2.807916e05, 3.333970e-01],
                [-7.713850e00, 2.545634e01, -9.592245e04, -1.792620e-01],
                [4.002520e21, -8.192010e21, 1.462011e23, -1.821224e20],
                [-2.912948e21, 6.399791e19, -3.531505e22, 1.964744e19],
                [-7.070723e19, 4.805948e19, 5.201014e21, -4.170396e17],
            ],
            [
                [-1.271181e00, 3.178340e-01, 5.830886e03, 7.186328e-03],
                [-7.090280e-01, 4.753706e-02, -1.757570e03, 1.465161e-03],
                [-8.686388e20, 9.428176e20, 1.738719e22, -1.251868e19],
                [1.877581e19, 1.097908e19, -1.006633e22, 2.734279e16],
                [2.238756e18, -7.688157e17, -2.061752e21, -3.427940e15],
            ],
        ]
    )
    p4e_o2o_vt_Cijk4: np.ndarray = np.array(
        [
            [
                [2.821759e00, 4.083138e00, -8.809991e01, 2.369644e-02],
                [3.841080e00, -9.986370e-01, -3.438479e01, 1.222381e-02],
                [4.330479e21, -1.677646e22, -5.573334e21, -3.089812e19],
                [-1.194045e20, -5.121704e19, 4.013656e21, -1.052730e18],
                [-9.939380e18, 4.180038e18, -2.265448e18, 1.050411e16],
            ],
            [
                [-1.066105e02, 6.618737e01, 2.630406e02, 2.791153e-02],
                [-2.658559e01, -9.167211e00, 1.678357e02, -1.064740e-01],
                [1.312884e22, -5.437653e21, 5.735816e21, 1.568233e20],
                [4.530951e21, 2.662341e20, -2.932068e22, 6.788371e18],
                [-3.473472e19, -5.623449e18, 2.765213e20, -6.030509e16],
            ],
            [
                [-3.476825e00, -4.156750e-01, 2.341590e01, -1.760866e-03],
                [-5.624110e-01, 1.663190e-01, 2.356659e00, -7.409818e-04],
                [1.908092e21, 1.107010e21, -1.769244e22, 1.272578e18],
                [-2.139241e19, 2.171483e19, -2.478535e19, 7.612186e16],
                [1.542813e18, -7.531694e17, -4.924709e18, -5.546602e14],
            ],
        ]
    )

    # ---------------- Basic helpers ---------------- #
    @staticmethod
    def p_max_electron_level(electron_energy: np.ndarray, ionization_potential: float, delta_E: float) -> int:
        idx = 0
        while idx + 1 < electron_energy.size and electron_energy[idx + 1] <= ionization_potential - delta_E:
            idx += 1
        return idx

    @staticmethod
    def p_Z_coll(T: float, n: float, coll_mass: float, diameter: float) -> float:
        return 4 * constants.K_CONST_PI * n * diameter * diameter * math.sqrt(constants.K_CONST_K * T / (2 * constants.K_CONST_PI * coll_mass))

    @staticmethod
    def integrate_semi_inf(func: Callable[[float], float], a: float = 0.0, subdivisions: int = constants.K_CONST_SUBDIVISIONS) -> float:
        return float(integrate_semi_inf_cpp(func, a=a, subdivisions=subdivisions))

    # ---------------- Partition functions and averages ---------------- #
    @staticmethod
    def p_Z_rot(T: float, rot_energy: np.ndarray, num_rot_levels: int, rot_symmetry: int) -> float:
        j = np.arange(num_rot_levels, dtype=float)
        return float(np.dot(2 * j + 1, np.exp(-rot_energy / (constants.K_CONST_K * T))) / rot_symmetry)

    @staticmethod
    def p_Z_vibr_eq(T: float, vibr_energy: np.ndarray) -> float:
        return float(np.sum(np.exp(-vibr_energy / (constants.K_CONST_K * T))))

    @staticmethod
    def p_Z_electron(T: float, electron_energy: np.ndarray, statistical_weight: np.ndarray, n_electron_levels: int) -> float:
        weights = statistical_weight[:n_electron_levels]
        energies = electron_energy[:n_electron_levels]
        return float(np.sum(weights * np.exp(-energies / (constants.K_CONST_K * T))))

    @staticmethod
    def p_avg_rot_energy(T: float, rot_energy: np.ndarray, num_rot_levels: int, rot_symmetry: int) -> float:
        j = np.arange(num_rot_levels, dtype=float)
        weights = (2 * j + 1) * rot_energy
        return float(
            np.dot(weights, np.exp(-rot_energy / (constants.K_CONST_K * T)))
            / (rot_symmetry * Approximation.p_Z_rot(T, rot_energy, num_rot_levels, rot_symmetry))
        )

    @staticmethod
    def p_avg_rot_energy_sq(T: float, rot_energy: np.ndarray, num_rot_levels: int, rot_symmetry: int) -> float:
        j = np.arange(num_rot_levels, dtype=float)
        weights = (2 * j + 1) * (rot_energy**2)
        return float(
            np.dot(weights, np.exp(-rot_energy / (constants.K_CONST_K * T)))
            / (rot_symmetry * Approximation.p_Z_rot(T, rot_energy, num_rot_levels, rot_symmetry))
        )

    # ---------------- Dissociation partition terms ---------------- #
    @staticmethod
    def Z_diss_U(
        T: float,
        U: float,
        electron_energy: np.ndarray,
        statistical_weight: np.ndarray,
        n_electron_levels: int,
        vibr_energy: Iterable[np.ndarray],
        i: int,
        e: int,
    ) -> float:
        res = 0.0
        for j in range(n_electron_levels):
            res += statistical_weight[j] * math.exp(electron_energy[j] / U) * Approximation.p_Z_vibr_eq(-U, vibr_energy[j]) / Approximation.p_Z_vibr_eq(T, vibr_energy[j])
        return (
            Approximation.p_Z_electron(T, electron_energy, statistical_weight, n_electron_levels)
            * math.exp((vibr_energy[e][i] + electron_energy[e]) * (1.0 / T + 1.0 / U) / constants.K_CONST_K)
            / res
        )

    @staticmethod
    def Z_diss_noU(
        T: float,
        electron_energy: np.ndarray,
        statistical_weight: np.ndarray,
        n_electron_levels: int,
        vibr_energy: Iterable[np.ndarray],
        num_vibr_levels: Iterable[int],
        i: int,
        e: int,
    ) -> float:
        res = 0.0
        for j in range(n_electron_levels):
            res += statistical_weight[j] * num_vibr_levels[j] / Approximation.p_Z_vibr_eq(T, vibr_energy[j])
        return (
            Approximation.p_Z_electron(T, electron_energy, statistical_weight, n_electron_levels)
            * math.exp((vibr_energy[e][i] + electron_energy[e]) / (constants.K_CONST_K * T))
            / res
        )

    @staticmethod
    def Z_diss_U_vibr(T: float, U: float, vibr_energy: np.ndarray, i: int) -> float:
        return Approximation.p_Z_vibr_eq(T, vibr_energy) * math.exp(vibr_energy[i] * (1.0 / T + 1.0 / U) / constants.K_CONST_K) / Approximation.p_Z_vibr_eq(-U, vibr_energy)

    @staticmethod
    def Z_diss_noU_vibr(T: float, vibr_energy: np.ndarray, num_vibr_levels: int, i: int) -> float:
        return Approximation.p_Z_vibr_eq(T, vibr_energy) * math.exp(vibr_energy[i] / (constants.K_CONST_K * T)) / num_vibr_levels

    def Z_diss(self, T: float, *args) -> float:
        if len(args) == 8:
            U, electron_energy, statistical_weight, n_electron_levels, vibr_energy, num_vibr_levels, i, e = args
            return self.Z_diss_U(T, U, electron_energy, statistical_weight, n_electron_levels, vibr_energy, i, e)
        if len(args) == 7:
            electron_energy, statistical_weight, n_electron_levels, vibr_energy, num_vibr_levels, i, e = args
            return self.Z_diss_noU(T, electron_energy, statistical_weight, n_electron_levels, vibr_energy, num_vibr_levels, i, e)
        if len(args) == 3:
            U, vibr_energy, i = args
            return self.Z_diss_U_vibr(T, U, vibr_energy, i)
        if len(args) == 4:
            vibr_energy, num_vibr_levels, i, _unused = args
            return self.Z_diss_noU_vibr(T, vibr_energy, num_vibr_levels, i)
        raise ModelParameterException("Unsupported Z_diss signature")

    # ---------------- Aliat coefficients ---------------- #
    @staticmethod
    def C_aliat(
        T: float,
        electron_energy: np.ndarray,
        statistical_weight: np.ndarray,
        n_electron_levels: int,
        vibr_energy: Iterable[np.ndarray],
        num_vibr_levels: Iterable[int],
        vibr_energy_product: float,
        activation_energy: float,
        i: int,
        e: int,
        U: float | None = None,
    ) -> float:
        kT = constants.K_CONST_K * T
        kU = constants.K_CONST_K * U if U is not None else math.inf
        res = 0.0
        for ee in range(n_electron_levels):
            tmp = 0.0
            for k in range(num_vibr_levels[ee]):
                i_e = vibr_energy[ee][k] + electron_energy[ee]
                if U is not None and i_e > activation_energy + vibr_energy_product:
                    tmp += statistical_weight[ee] * math.exp((activation_energy + vibr_energy_product - i_e) / kT)
                elif U is not None:
                    tmp += statistical_weight[ee] * math.exp(-(activation_energy + vibr_energy_product - i_e) / kU)
                else:
                    tmp += statistical_weight[ee] * (math.exp((activation_energy + vibr_energy_product - i_e) / kT) if i_e > activation_energy + vibr_energy_product else 1.0)
            res += tmp / Approximation.p_Z_vibr_eq(T, vibr_energy[ee])
        res = 1.0 / res
        i_e = vibr_energy[e][i] + electron_energy[e]
        if U is not None:
            if i_e > activation_energy + vibr_energy_product:
                res *= math.exp((activation_energy + vibr_energy_product) / kT)
            else:
                res *= math.exp(-(activation_energy + vibr_energy_product) / kU) * math.exp(i_e * (1.0 / kT + 1.0 / kU))
        else:
            res *= math.exp((activation_energy + vibr_energy_product) / kT if i_e > activation_energy + vibr_energy_product else i_e / kT)
        return res * Approximation.p_Z_electron(T, electron_energy, statistical_weight, n_electron_levels)

    # ---------------- Vibrational ladder conversions ---------------- #
    @staticmethod
    def convert_vibr_ladder_N2(vibr_energy: float) -> float:
        te = 1000.0 * vibr_energy / constants.K_CONST_EV
        return (
            0.58142
            - 0.0039784 * te
            + 1.3127e-5 * te * te
            - 9.877e-9 * te * te * te
            + 3.8664e-12 * te * te * te * te
            - 8.4268e-16 * te**5
            + 1.0326e-19 * te**6
            - 6.649e-24 * te**7
            + 1.7506e-28 * te**8
        )

    @staticmethod
    def convert_vibr_ladder_O2(vibr_energy: float) -> float:
        te = vibr_energy / constants.K_CONST_EV
        return (
            0.4711499685975582
            + 2.23383944 * te
            + 2.17784653 * te * te
            + 0.61279172 * te**3
            - 1.38809821 * te**4
            + 0.66364689 * te**5
            - 0.13147542 * te**6
            + 0.00950554 * te**7
        )

    # ---------------- Velocity helpers ---------------- #
    @staticmethod
    def vel_avg_vv(rel_vel: float, coll_mass: float, delta_E_vibr: float) -> float:
        rel_vel_after_sq = delta_E_vibr * (2.0 / coll_mass) + rel_vel * rel_vel
        if rel_vel_after_sq < 0:
            return -1.0
        return 0.5 * (rel_vel + math.sqrt(rel_vel_after_sq))

    @staticmethod
    def vel_avg_vt(rel_vel: float, coll_mass: float, delta_E_vibr: float) -> float:
        rel_vel_after_sq = rel_vel * rel_vel - delta_E_vibr * (2.0 / coll_mass)
        if rel_vel_after_sq < 0:
            return -1.0
        return 0.5 * (rel_vel + math.sqrt(rel_vel_after_sq))

    @staticmethod
    def min_vel_diss(coll_mass: float, diss_energy: float, vibr_energy: float) -> float:
        if diss_energy <= vibr_energy:
            return 0.0
        return math.sqrt(2 * (diss_energy - vibr_energy) / coll_mass)

    @staticmethod
    def min_vel_diss_ILT_N2N(coll_mass: float, vibr_energy: float, i: float) -> float:
        if i <= 8:
            c1 = 1.786e-18
        elif i <= 34:
            c1 = 1.71e-18
        elif i <= 52:
            c1 = 1.68e-18
        else:
            c1 = 1.66e-18
        return math.sqrt(2 * (c1 - vibr_energy) / coll_mass)

    @staticmethod
    def min_vel_diss_ILT_O2O(coll_mass: float, i: float) -> float:
        c1 = 0.3867 * i * i * i - 2.7425 * i * i - 1901.9 * i + 61696
        return math.sqrt(2 * c1 * constants.K_CONST_K / coll_mass)

    # ---------------- Probability models (FHO) ---------------- #
    @staticmethod
    def probability_VV_FHO(rel_vel: float, coll_mass: float, delta_E_vibr: float, i: int, k: int, delta_i: int, omega1: float, omega2: float, omega: float, alpha_FHO: float) -> float:
        ns12 = (Approximation.factorial_table[i + delta_i] / Approximation.factorial_table[i]) * (Approximation.factorial_table[k] / Approximation.factorial_table[k - delta_i])
        eps = abs(delta_E_vibr) / (constants.K_CONST_K * rel_vel * rel_vel * coll_mass * 0.5)
        s = (omega1 + omega2) / omega
        return ns12 * pow(eps, s) * math.exp(-2.0 * pow(ns12, 1.0 / s) * eps / (s + 1.0)) / (Approximation.factorial_table[int(s)] * Approximation.factorial_table[int(s)])

    @staticmethod
    def probability_VT_FHO(rel_vel: float, coll_mass: float, osc_mass: float, delta_E_vibr: float, i: int, delta_i: int, omega: float, ram: float, alpha_FHO: float, E_FHO: float, svt_FHO: float) -> float:
        eps = abs(delta_E_vibr) / (constants.K_CONST_K * rel_vel * rel_vel * coll_mass * 0.5)
        s = (ram + 1) / omega
        ns = Approximation.factorial_table[i + delta_i] / Approximation.factorial_table[i] if delta_i >= 0 else Approximation.factorial_table[i] / Approximation.factorial_table[i + delta_i]
        return ns * pow(eps, s) * math.exp(-2.0 * pow(ns, 1.0 / s) * eps / (s + 1.0)) / (Approximation.factorial_table[int(s)] * Approximation.factorial_table[int(s)])

    @staticmethod
    def p_probability_diss(rel_vel: float, coll_mass: float, diss_energy: float, vibr_energy: float, center_of_mass: bool = True) -> float:
        energy = vibr_energy + 0.5 * coll_mass * rel_vel * rel_vel
        if energy < diss_energy:
            return 0.0
        if center_of_mass:
            return 1.0 - diss_energy / energy
        return 1.0

    # ---------------- Cross sections (RS/VSS) ---------------- #
    @staticmethod
    def crosssection_elastic_RS(diameter: float) -> float:
        return constants.K_CONST_PI * diameter * diameter

    @staticmethod
    def crosssection_elastic_VSS(rel_vel: float, coll_mass: float, vss_c: float, vss_omega: float) -> float:
        return constants.K_CONST_PI * pow(vss_c, 2) * pow(rel_vel, 1 - 2 * vss_omega)

    @staticmethod
    def crosssection_elastic_VSS_cs(rel_vel: float, vss_c_cs: float, vss_omega: float) -> float:
        return vss_c_cs * pow(rel_vel, 1 - 2 * vss_omega)

    @staticmethod
    def crosssection_VT_FHO_RS(
        rel_vel: float,
        coll_mass: float,
        diameter: float,
        osc_mass: float,
        delta_E_vibr: float,
        i: int,
        delta_i: int,
        omega: float,
        ram: float,
        alpha_FHO: float,
        E_FHO: float,
        svt_FHO: float,
    ) -> float:
        prob = Approximation.probability_VT_FHO(rel_vel, coll_mass, osc_mass, delta_E_vibr, i, delta_i, omega, ram, alpha_FHO, E_FHO, svt_FHO)
        return Approximation.crosssection_elastic_RS(diameter) * prob

    @staticmethod
    def crosssection_VT_FHO_VSS(
        rel_vel: float,
        coll_mass: float,
        vss_c_cs: float,
        vss_omega: float,
        osc_mass: float,
        delta_E_vibr: float,
        i: int,
        delta_i: int,
        omega: float,
        ram: float,
        alpha_FHO: float,
        E_FHO: float,
        svt_FHO: float,
    ) -> float:
        prob = Approximation.probability_VT_FHO(rel_vel, coll_mass, osc_mass, delta_E_vibr, i, delta_i, omega, ram, alpha_FHO, E_FHO, svt_FHO)
        return Approximation.crosssection_elastic_VSS_cs(rel_vel, vss_c_cs, vss_omega) * prob

    @staticmethod
    def crosssection_diss_RS(rel_vel: float, coll_mass: float, diameter: float, diss_energy: float, vibr_energy: float, center_of_mass: bool = True) -> float:
        prob = Approximation.p_probability_diss(rel_vel, coll_mass, diss_energy, vibr_energy, center_of_mass)
        return prob * Approximation.crosssection_elastic_RS(diameter)

    @staticmethod
    def crosssection_diss_VSS(rel_vel: float, coll_mass: float, vss_c_cs: float, vss_omega: float, diss_energy: float, vibr_energy: float, center_of_mass: bool = True) -> float:
        prob = Approximation.p_probability_diss(rel_vel, coll_mass, diss_energy, vibr_energy, center_of_mass)
        return prob * Approximation.crosssection_elastic_VSS_cs(rel_vel, vss_c_cs, vss_omega)

    # ---------------- Debye length ---------------- #
    @staticmethod
    def debye_length(T: float, concentrations: np.ndarray, charges: np.ndarray) -> float:
        denom = float(np.dot(concentrations, charges * charges))
        if denom <= 0:
            raise ValueError("Debye length denominator must be positive")
        return math.sqrt(constants.K_CONST_E0 * constants.K_CONST_K * T / denom)

    @staticmethod
    def max_electron_level(atom, delta_E: float) -> int:
        return Approximation.p_max_electron_level(atom.electron_energy, atom.ionization_potential, delta_E)

    # ---------------- Partition helpers using Molecule data ---------------- #
    def Z_rot(self, T: float, molecule: Molecule, i: int, e: int) -> float:
        return self.p_Z_rot(T, molecule.rot_energy[e][i], molecule.num_rot_levels[e][i], molecule.rot_symmetry)

    def Z_vibr_eq(self, T: float, molecule: Molecule, e: int) -> float:
        return self.p_Z_vibr_eq(T, molecule.vibr_energy[e])

    def Z_electron(self, T: float, molecule: Molecule, num_electron_levels: int) -> float:
        if num_electron_levels == -1:
            return self.p_Z_electron(T, molecule.electron_energy, molecule.statistical_weight, molecule.num_electron_levels)
        return self.p_Z_electron(T, molecule.electron_energy, molecule.statistical_weight, num_electron_levels)

    def avg_rot_energy(self, T: float, molecule: Molecule, i: int, e: int) -> float:
        return self.p_avg_rot_energy(T, molecule.rot_energy[e][i], molecule.num_rot_levels[e][i], molecule.rot_symmetry)

    def avg_rot_energy_sq(self, T: float, molecule: Molecule, i: int, e: int) -> float:
        return self.p_avg_rot_energy_sq(T, molecule.rot_energy[e][i], molecule.num_rot_levels[e][i], molecule.rot_symmetry)

    def c_vibr_approx(self, T: float, molecule: Molecule) -> float:
        # Simple harmonic approximation for vibrational specific heat per mass
        res = 0.0
        for e in range(molecule.num_electron_levels):
            theta_v = molecule.characteristic_vibr_temperatures[e] if molecule.characteristic_vibr_temperatures.size > e else 0.0
            if theta_v > 0:
                x = theta_v / T
                res += constants.K_CONST_K * (x * x * math.exp(x)) / ((math.exp(x) - 1) ** 2)
        return res / molecule.mass

    # ---------------- Treanor distribution helper ---------------- #
    @staticmethod
    def p_max_i(T: float, vibr_energy1: float, vibr_frequency: float, alpha: float) -> int:
        # Eq. 1.6 in Kustova & Nagnibeda (approximation used in C++ code)
        i_star = int((vibr_energy1 / (constants.K_CONST_K * T) - 1) / (2 * alpha))
        return max(i_star, 0)

    def max_i(self, T: float, molecule: Molecule) -> int:
        if not molecule.anharmonic_spectrum:
            return molecule.num_vibr_levels[0] if molecule.num_vibr_levels else 0
        i_star = self.p_max_i(
            T,
            molecule.vibr_energy[0][1] - molecule.vibr_energy[0][0],
            constants.K_CONST_C * molecule.vibr_frequency[0],
            molecule.vibr_we_xe[0] / molecule.vibr_frequency[0],
        )
        if i_star < 1:
            raise ValueError("No Treanor distribution possible for these T/T1 values")
        if molecule.num_vibr_levels and i_star >= molecule.num_vibr_levels[0] - 1:
            return molecule.num_vibr_levels[0] - 1
        return i_star

    # ---------------- Specific heats ---------------- #
    @staticmethod
    def p_c_tr(T: float, mass: float) -> float:
        return 1.5 * constants.K_CONST_K / mass

    @staticmethod
    def p_c_rot(T: float, mass: float, rot_energy: np.ndarray, num_rot_levels: int, rot_symmetry: int) -> float:
        avg_E = Approximation.p_avg_rot_energy(T, rot_energy, num_rot_levels, rot_symmetry)
        avg_E_sq = Approximation.p_avg_rot_energy_sq(T, rot_energy, num_rot_levels, rot_symmetry)
        return (avg_E_sq - avg_E * avg_E) / (constants.K_CONST_K * T * T * mass)

    def c_tr(self, T: float, particle: Particle) -> float:
        return self.p_c_tr(T, particle.mass)

    def c_rot(self, temperature: float, molecule: Molecule, vibr_level: int = 0, e_level: int = 0) -> float:
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        if e_level < 0 or e_level >= molecule.num_electron_levels:
            raise ValueError("Invalid electronic level index")
        if vibr_level < 0 or (molecule.num_vibr_levels and vibr_level >= molecule.num_vibr_levels[e_level]):
            raise ValueError("Invalid vibrational level index")
        rot_energy = molecule.rot_energy[e_level][vibr_level]
        num_rot_levels = molecule.num_rot_levels[e_level][vibr_level]
        return self.p_c_rot(temperature, molecule.mass, rot_energy, num_rot_levels, molecule.rot_symmetry)

    # ---------------- Omega integrals ---------------- #
    @staticmethod
    def omega_integral_RS(T: float, l: int, r: int, diameter: float, coll_mass: float) -> float:
        return (
            math.sqrt(T * constants.K_CONST_K / (2 * constants.K_CONST_PI * coll_mass))
            * 0.5
            * Approximation.factorial_table[r + 1]
            * (1.0 - 0.5 * (1.0 + (-1.0) ** l) / (l + 1))
            * constants.K_CONST_PI
            * (diameter * diameter)
        )

    @staticmethod
    def omega_integral_VSS(T: float, l: int, r: int, coll_mass: float, vss_c_cs: float, vss_omega: float, vss_alpha: float) -> float:
        # multiplicative factor depending on l
        if l == 1:
            mult = 1.0 / (1.0 + vss_alpha)
        elif l == 2:
            mult = 2 * vss_alpha / ((2 + vss_alpha) * (1 + vss_alpha))
        elif l == 3:
            mult = (3 * vss_alpha * vss_alpha + vss_alpha + 2) / ((3 + vss_alpha) * (2 + vss_alpha) * (1 + vss_alpha))
        elif l == 4:
            mult = (4 * vss_alpha**3 + 12 * vss_alpha * vss_alpha + 32 * vss_alpha) / ((4 + vss_alpha) * (3 + vss_alpha) * (2 + vss_alpha) * (1 + vss_alpha))
        elif l == 5:
            mult = (
                5 * vss_alpha**4 + 30 * vss_alpha**3 + 115 * vss_alpha * vss_alpha + 90 * vss_alpha + 120
            ) / ((5 + vss_alpha) * (4 + vss_alpha) * (3 + vss_alpha) * (2 + vss_alpha) * (1 + vss_alpha))
        else:
            raise ValueError(f"Unsupported l={l} for VSS omega integral")

        return mult * math.sqrt(T * constants.K_CONST_K / (2 * constants.K_CONST_PI * coll_mass)) * special.gamma(2.5 + r - vss_omega) * vss_c_cs * pow(
            2 * T * constants.K_CONST_K / coll_mass, 0.5 - vss_omega
        )

    @staticmethod
    def omega_integral_LennardJones(T: float, l: int, r: int, epsilon: float) -> float:
        if (r, l) == (1, 1):
            log_KT_eps = math.log(constants.K_CONST_K * T / epsilon) + 1.4
            return 1.0 / (
                -0.16845
                - 0.02258 / (log_KT_eps * log_KT_eps)
                + 0.19779 / log_KT_eps
                + 0.64373 * log_KT_eps
                - 0.09267 * log_KT_eps * log_KT_eps
                + 0.00711 * log_KT_eps * log_KT_eps * log_KT_eps
            )
        if (r, l) == (2, 2):
            log_KT_eps = math.log(constants.K_CONST_K * T / epsilon) + 1.5
            return 1.0 / (
                -0.40811
                - 0.05086 / (log_KT_eps * log_KT_eps)
                + 0.34010 / log_KT_eps
                + 0.70375 * log_KT_eps
                - 0.10699 * log_KT_eps * log_KT_eps
                + 0.00763 * log_KT_eps * log_KT_eps * log_KT_eps
            )
        raise ModelParameterException("Lennard-Jones omega integral only implemented for (l,r) in {(1,1),(2,2)}")

    @staticmethod
    def omega_integral_Born_Mayer(T: float, l: int, r: int, beta: float, phi_zero: float, diameter: float, epsilon: float) -> float:
        if not ((r == 1 and l == 1) or (r == 2 and l == 2)):
            raise ModelParameterException("Born-Mayer omega integral only implemented for (1,1) and (2,2)")
        log_v_star = math.log(phi_zero / (10 * epsilon))
        KT_eps = constants.K_CONST_K * T / epsilon
        # coeffs from C++ Born_Mayer_coeff_array
        Born_Mayer_coeff_array = np.array(
            [
                [1.5931e-2, -3.4482e-2, -2.3367e-2, -2.5497e-2],
                [3.0327e-2, -1.4465e-1, -4.1806e-2, 2.1043e-2],
                [2.9699e-2, -1.0849e-1, -4.4383e-2, 1.9486e-2],
                [1.4647e-2, -4.9057e-2, -3.6874e-2, 1.6506e-2],
                [2.3127e-2, -1.2844e-1, -5.0611e-2, 2.6622e-2],
                [1.6141e-2, -4.4034e-2, -2.6205e-2, 1.5382e-2],
            ]
        )
        A = np.zeros(6)
        for i in range(6):
            A[i] = Born_Mayer_coeff_array[i, 0] + pow(log_v_star / (beta * diameter), -2) * (
                Born_Mayer_coeff_array[i, 1] + Born_Mayer_coeff_array[i, 2] / log_v_star + Born_Mayer_coeff_array[i, 3] / (log_v_star * log_v_star)
            )
        if (r, l) == (1, 1):
            return pow(math.log(phi_zero / (constants.K_CONST_K * T)) / (beta * diameter), 2) * (
                0.89 + A[0] / (KT_eps * KT_eps) + A[1] / (KT_eps**4) + A[2] / (KT_eps**6)
            )
        # (r,l)==(2,2)
        log_KT_eps = math.log(KT_eps)
        return pow(math.log(phi_zero / (constants.K_CONST_K * T)) / (beta * diameter), 2) * (
            1.04 + A[3] / (log_KT_eps * log_KT_eps) + A[4] / (log_KT_eps**3) + A[5] / (log_KT_eps**4)
        )

    def _omega_integral_bracket(
        self,
        T: float,
        l: int,
        r: int,
        base: Callable[[float, int, int], float],
        interaction: Interaction,
        dimensional: bool,
    ) -> float:
        step = constants.K_CONST_OMEGA_D_STEP_SIZE

        def rs(temp: float, l_val: int, r_val: int) -> float:
            return self.omega_integral_RS(temp, l_val, r_val, interaction.collision_diameter, interaction.collision_mass)

        def base_scaled(temp: float) -> float:
            return base(temp, l, l) * rs(temp, l, l)

        if (l, r) in ((1, 1), (2, 2)):
            val = base(T, l, r)
            return val * rs(T, l, r) if dimensional else val

        if (l, r) in ((1, 2), (2, 3)):
            res = ((base_scaled(T + step) - base_scaled(T - step)) * T / (2 * step)) + (l + 1.5) * base_scaled(T)
            return res if dimensional else res / rs(T, l, r)

        if (l, r) in ((1, 3), (2, 4)):
            r0 = base_scaled(T)
            rp1 = base_scaled(T + step)
            rm1 = base_scaled(T - step)
            res = (T * T * (rp1 - 2 * r0 + rm1) / (step * step)) + (T * (2 * l + 5) * (rp1 - rm1) / (2 * step)) + (l + 2.5) * (l + 1.5) * r0
            return res if dimensional else res / rs(T, l, r)

        if l == 1 and r == 4:
            r0 = base_scaled(T)
            rp1 = base_scaled(T + step)
            rm1 = base_scaled(T - step)
            rp2 = base_scaled(T + 2 * step)
            rm2 = base_scaled(T - 2 * step)
            res = (
                T * T * T * (rp2 - 2 * rp1 + 2 * rm1 - rm2) / (2 * step * step * step)
                + T * T * (3 * l + 14.5) * (rp1 - 2 * r0 + rm1) / (step * step)
                + T * ((2 * l + 9) * (l + 4.5) + (l + 2.5) * (l + 1.5)) * (rp1 - rm1) / (2 * step)
                + (l + 3.5) * (l + 2.5) * (l + 1.5) * r0
            )
            return res if dimensional else res / rs(T, l, r)

        if l == 1 and r == 5:
            prev = self._omega_integral_bracket(T, 1, 4, base, interaction, True)
            res = (self._omega_integral_bracket(T + step, 1, 4, base, interaction, True) - self._omega_integral_bracket(T - step, 1, 4, base, interaction, True)) * T / (
                2 * step
            ) + 5.5 * prev
            return res if dimensional else res / rs(T, l, r)

        raise ValueError(f"Unsupported (l,r)=({l},{r}) for bracket integral")

    @staticmethod
    def omega_integral_ESA_hydrogen(T: float, diameter: float, a1: float, a2: float, a3: float, a4: float, a5: float, a6: float, a7: float) -> float:
        x = math.log(T)
        exp_1 = math.exp((x - a3) / a4)
        exp_2 = math.exp((x - a6) / a7)
        return ((a1 + a2 * x) * exp_1 / (exp_1 + 1 / exp_1) + a5 * exp_2 / (exp_2 + 1 / exp_2)) / (diameter * diameter * 1e20)

    def omega_integral_ESA_H2H2(self, T: float, l: int, r: int, diameter: float) -> float:
        coeffs = {
            (1, 1): (24.00841090, -1.61027553, 3.88885724, -8.89043396, 0.44260972, 8.88408687, -1.05402226),
            (1, 2): (23.02146328, -1.70509850, 3.88885724, -10.46929121, 0.36330166, 8.26405726, -1.02331842),
            (1, 3): (21.17218602, -1.57714612, 3.88885724, -9.72209606, 0.59112956, 8.15580488, -1.46063978),
            (1, 4): (20.05416161, -1.51326919, 3.88885724, -9.38278743, 0.70004430, 8.00952510, -1.57063623),
            (1, 5): (19.06639058, -1.45577823, 3.88885724, -9.14716131, 0.81250655, 7.85268967, -1.66995743),
            (2, 2): (27.54387526, -1.98253166, 3.88885724, -12.91940775, 0.34707960, 8.72131306, -0.88296275),
            (2, 3): (26.22527642, -1.94538819, 3.88885724, -13.40557645, 0.40398208, 8.42662474, -0.96878644),
            (2, 4): (24.59185702, -1.83729737, 3.88885724, -12.78050876, 0.62739891, 8.27557505, -1.33071440),
            (3, 3): (24.57128293, -1.80855250, 3.88885724, -11.86035430, 0.36590658, 8.38682707, -1.00746362),
        }
        key = (l, r)
        if key not in coeffs:
            raise ModelParameterException("ESA_H2H2 omega integral not available for this (l,r)")
        return self.omega_integral_ESA_hydrogen(T, diameter, *coeffs[key])

    def omega_integral_ESA_H2H(self, T: float, l: int, r: int, diameter: float) -> float:
        coeffs = {
            (1, 1): (12.49063970, -1.14704753, 8.76515503, -3.52921496, 0.32874932, 12.77040465, -3.04802967),
            (1, 2): (12.02124035, -1.19514025, 8.76515503, -3.45192920, 0.45922882, 12.77040465, -2.29080329),
            (1, 3): (11.69204285, -1.24240232, 8.76515503, -3.49608019, 0.63354264, 12.77040465, -2.29080329),
            (1, 4): (11.45792771, -1.29677120, 8.76515503, -3.64478512, 0.85298582, 12.77040465, -2.29080329),
            (1, 5): (11.00483923, -1.27212994, 8.76515503, -3.51537463, 0.85298582, 12.77040465, -2.29080329),
            (2, 2): (7.45048892, -1.43326160, 9.59201391, -1.35066206, 7.15859874, 9.88881724, -1.39484886),
            (2, 3): (10.84507417, -1.42859529, 9.20889644, -1.29890434, 3.37747184, 9.83307970, -1.30321649),
            (2, 4): (11.55088396, -1.41480945, 8.98739895, -1.39880703, 2.32276221, 9.89142509, -1.26804718),
            (3, 3): (-15.25288758, -1.39293852, 9.59147724, -1.62599901, 28.71128123, 9.68396961, -1.63186985),
        }
        key = (l, r)
        if key not in coeffs:
            raise ModelParameterException("ESA_H2H omega integral not available for this (l,r)")
        return self.omega_integral_ESA_hydrogen(T, diameter, *coeffs[key])

    def omega_integral_ESA_HH(self, T: float, l: int, r: int, diameter: float) -> float:
        coeffs = {
            (1, 1): (15.09506044, -1.25710008, 9.57839369, -3.80371463, 0.98646613, 9.25705877, -0.93611707),
            (1, 2): (14.14566908, -1.17057105, 9.02830724, -3.00779776, 0.74653903, 9.10299040, -0.68184353),
            (1, 3): (13.39722075, -1.09886403, 8.50097335, -2.86025395, 0.85345727, 8.90666490, -0.67571329),
            (1, 4): (12.97073246, -1.06479185, 8.18885522, -2.78105132, 0.89401865, 8.73403138, -0.65658782),
            (1, 5): (12.69248000, -1.04857945, 7.97861283, -2.73621289, 0.90816787, 8.57840253, -0.63732002),
            (2, 2): (22.08948804, -1.85066626, 8.50932055, -7.66943974, 0.77454531, 9.69545318, -0.62104466),
            (2, 3): (17.94703897, -1.42488999, 7.66669340, -4.76239721, 1.26783524, 9.53716768, -0.73914215),
            (2, 4): (18.78590499, -1.59291967, 7.97734302, -5.66814860, 1.01816360, 9.32328437, -0.60882006),
            (3, 3): (13.82986524, -1.01454290, 7.48970759, -3.27628187, 2.08225623, 9.21388055, -1.32086596),
        }
        key = (l, r)
        if key not in coeffs:
            raise ModelParameterException("ESA_HH omega integral not available for this (l,r)")
        return self.omega_integral_ESA_hydrogen(T, diameter, *coeffs[key])

    @staticmethod
    def _esa_eval(a: np.ndarray, x: float) -> float:
        exp_1 = math.exp((x - a[2]) / a[3])
        exp_2 = math.exp((x - a[5]) / a[6])
        return math.exp((a[0] + a[1] * x) * exp_1 / (exp_1 + 1 / exp_1) + a[4] * exp_2 / (exp_2 + 1 / exp_2))

    def omega_integral_ESA_nn(self, T: float, l: int, r: int, beta: float, epsilon_zero: float, r_e: float) -> float:
        x = math.log(constants.K_CONST_K * T / epsilon_zero)
        coeffs: dict[tuple[int, int], np.ndarray] = {
            (1, 1): np.array(
                [
                    (7.884756e-1) - (2.438494e-2) * beta,
                    -2.952759e-1 - (1.744149e-3) * beta,
                    (5.020892e-1) + (4.316985e-2) * beta,
                    (-9.042460e-1) - (4.017103e-2) * beta,
                    (-3.373058) + (2.458538e-1) * beta - (4.850047e-3) * beta * beta,
                    (4.161981) + (2.202737e-1) * beta - (1.718010e-2) * beta * beta,
                    (2.462523) + (3.231308e-1) * beta - (2.281072e-2) * beta * beta,
                ]
            ),
            (1, 2): np.array(
                [
                    (7.123565e-1) + (-2.688875e-2) * beta,
                    (-2.9105e-1) + (-2.065175e-3) * beta,
                    (4.187065e-2) + (4.060236e-2) * beta,
                    (-9.287685e-1) + (-2.342270e-2) * beta,
                    (-3.598542) + (2.54512e-1) * beta + (-4.685966e-3) * beta * beta,
                    (3.934824) + (2.699944e-1) * beta + (-2.009886e-2) * beta * beta,
                    2.578084 + (3.449024e-1) * beta + (-2.2927e-2) * beta * beta,
                ]
            ),
            (1, 3): np.array(
                [
                    (6.606022e-1) + (-2.831448e-2) * beta,
                    (-2.8709e-1) + (-2.232827e-3) * beta,
                    (-2.51969e-1) + (3.778211e-2) * beta,
                    (-9.173046e-1) + (-1.864476e-2) * beta,
                    (-3.776812) + (2.552528e-1) * beta + (-4.23722e-3) * beta * beta,
                    (3.768103) + (3.155025e-1) * beta + (-2.218849e-2) * beta * beta,
                    2.695440 + (3.597998e-1) * beta + (-2.267102e-2) * beta * beta,
                ]
            ),
            (1, 4): np.array(
                [
                    (6.268016e-1) + (-2.945078e-2) * beta,
                    (-2.830834e-1) + (-2.361273e-3) * beta,
                    (-4.559927e-1) + (3.705640e-2) * beta,
                    (-9.334638e-1) + (-1.797329e-2) * beta,
                    (-3.947019) + (2.446843e-1) * beta + (-3.176374e-3) * beta * beta,
                    (3.629926) + (3.761272e-1) * beta + (-2.451016e-2) * beta * beta,
                    (2.824905) + (3.781709e-1) * beta + (-2.251978e-2) * beta * beta,
                ]
            ),
            (1, 5): np.array(
                [
                    (5.956859e-1) + (-2.915893e-2) * beta,
                    (-2.804989e-1) + (-2.298968e-3) * beta,
                    (-5.965551e-1) + (3.724395e-2) * beta,
                    (-8.946001e-1) + (-2.550731e-2) * beta,
                    (-4.076798) + (1.983892e-1) * beta + (-5.014065e-3) * beta * beta,
                    (3.458362) + (4.770695e-1) * beta + (-2.678054e-2) * beta * beta,
                    2.982260 + (4.014572e-1) * beta + (-2.142580e-2) * beta * beta,
                ]
            ),
            (2, 2): np.array(
                [
                    (7.898524e-1) - (2.114115e-2) * beta,
                    (-2.998325e-1) - (1.243977e-3) * beta,
                    (7.077103e-1) + (3.583907e-2) * beta,
                    (-8.946857e-1) - (2.473947e-2) * beta,
                    (-2.958969) + (2.303358e-1) * beta - (5.226562e-3) * beta * beta,
                    (4.348412) + (1.920321e-1) * beta - (1.496557e-2) * beta * beta,
                    (2.205440) + (2.567027e-1) * beta - (1.861359e-2) * beta * beta,
                ]
            ),
            (2, 3): np.array(
                [
                    (7.269006e-1) - (2.233866e-2) * beta,
                    (-2.972304e-1) - (1.392888e-3) * beta,
                    (3.904230e-1) + (3.231655e-2) * beta,
                    (-9.442201e-1) - (1.494805e-2) * beta,
                    (-3.137828) + (2.347767e-1) * beta - (4.963979e-3) * beta * beta,
                    (4.190370) + (2.346004e-1) * beta - (1.718963e-2) * beta * beta,
                    (2.319751) + (2.700236e-1) * beta - (1.854217e-2) * beta * beta,
                ]
            ),
            (2, 4): np.array(
                [
                    (6.829159e-1) - (2.233866e-2) * beta,
                    (-2.943232e-1) - (1.514322e-3) * beta,
                    (1.414623e-1) + (3.075351e-2) * beta,
                    (-9.720228e-1) - (1.038869e-2) * beta,
                    (-3.284219) + (2.243767e-1) * beta - (3.913041e-3) * beta * beta,
                    (4.011692) + (3.005083e-1) * beta - (2.012373e-2) * beta * beta,
                    (2.401249) + (2.943600e-1) * beta - (1.884503e-2) * beta * beta,
                ]
            ),
            (3, 3): np.array(
                [
                    (7.468781e-1) - (2.518134e-2) * beta,
                    (-2.947438e-1) - (1.811571e-3) * beta,
                    (2.234096e-1) + (3.681114e-2) * beta,
                    (-9.974591e-1) - (2.670805e-2) * beta,
                    (-3.381787) + (2.372932e-1) * beta - (4.239629e-3) * beta * beta,
                    (4.094540) + (2.756466e-1) * beta - (2.009227e-2) * beta * beta,
                    (2.476087) + (3.300898e-1) * beta - (2.223317e-2) * beta * beta,
                ]
            ),
            (4, 4): np.array(
                [
                    (7.365470e-1) - (2.242357e-2) * beta,
                    (-2.968650e-1) - (1.396696e-3) * beta,
                    (3.747555e-1) + (2.847063e-2) * beta,
                    (-9.944036e-1) - (1.378926e-2) * beta,
                    (-3.136655) + (2.176409e-1) * beta - (3.899247e-3) * beta * beta,
                    (4.145871) + (2.855836e-1) * beta - (1.939452e-2) * beta * beta,
                    (2.315532) + (2.842981e-1) * beta - (1.874462e-2) * beta * beta,
                ]
            ),
        }
        key = (l, r)
        if key not in coeffs:
            raise ModelParameterException("ESA_nn omega integral not available for this (l,r)")
        return self._esa_eval(coeffs[key], x)

    def omega_integral_ESA_cn(self, T: float, l: int, r: int, beta: float, epsilon_zero: float, r_e: float) -> float:
        x = math.log(constants.K_CONST_K * T / epsilon_zero)
        coeffs: dict[tuple[int, int], np.ndarray] = {
            (1, 1): np.array(
                [
                    (9.851755e-1) - (2.870704e-2) * beta,
                    -(4.737800e-1) - (1.370344e-3) * beta,
                    (7.080799e-1) + (4.575312e-3) * beta,
                    (-1.239439) - (3.683605e-2) * beta,
                    (-4.638467) + (4.418904e-1) * beta - (1.220292e-2) * beta * beta,
                    (3.841835) + (3.277658e-1) * beta - (2.660275e-2) * beta * beta,
                    (2.317342) + (3.912768e-1) * beta - (3.136223e-2) * beta * beta,
                ]
            ),
            (1, 2): np.array(
                [
                    (8.361751e-1) - (3.201292e-2) * beta,
                    -(4.707355e-1) - (1.783284e-3) * beta,
                    (1.771157e-1) + (1.172773e-2) * beta,
                    (-1.094937) - (3.115598e-2) * beta,
                    (-4.976384) + (4.708074e-1) * beta - (1.283818e-2) * beta * beta,
                    (3.645873) + (3.699452e-1) * beta - (2.988684e-2) * beta * beta,
                    (2.428864) + (4.267351e-1) * beta - (3.278874e-2) * beta * beta,
                ]
            ),
            (1, 3): np.array(
                [
                    (7.440562e-1) - (3.453851e-2) * beta,
                    -(4.656306e-1) - (2.097901e-3) * beta,
                    (-1.465752e-1) + (1.446209e-2) * beta,
                    (-1.080410) - (2.712029e-2) * beta,
                    (-5.233907) + (4.846691e-1) * beta - (1.280346e-2) * beta * beta,
                    (3.489814) + (4.140270e-1) * beta - (3.250138e-2) * beta * beta,
                    (2.529678) + (4.515088e-1) * beta - (3.339293e-2) * beta * beta,
                ]
            ),
            (1, 4): np.array(
                [
                    (6.684360e-1) - (3.515695e-2) * beta,
                    -(4.622014e-1) - (2.135808e-3) * beta,
                    (-3.464990e-1) + (1.336362e-2) * beta,
                    (-1.054374) - (3.149321e-2) * beta,
                    (-5.465789) + (4.888443e-1) * beta - (1.228090e-2) * beta * beta,
                    (3.374614) + (4.602468e-1) * beta - (3.463073e-2) * beta * beta,
                    (2.648622) + (4.677409e-1) * beta - (3.339297e-2) * beta * beta,
                ]
            ),
            (1, 5): np.array(
                [
                    (6.299083e-1) - (3.720000e-2) * beta,
                    -(4.560921e-1) - (2.395779e-3) * beta,
                    (-5.228598e-1) + (1.594610e-2) * beta,
                    (-1.124725) - (2.862354e-2) * beta,
                    (-5.687354) + (4.714789e-1) * beta - (1.056602e-2) * beta * beta,
                    (3.267709) + (5.281419e-1) * beta - (3.678869e-2) * beta * beta,
                    (2.784725) + (4.840700e-1) * beta - (3.265127e-2) * beta * beta,
                ]
            ),
            (2, 2): np.array(
                [
                    (9.124518e-1) + (-2.398461e-2) * beta,
                    (-4.697184e-1) + (-7.809681e-4) * beta,
                    (1.031053) + (4.069668e-3) * beta,
                    (-1.090782) + (-2.413508e-2) * beta,
                    (-4.127243) + (4.302667e-1) * beta + (-1.352874e-2) * beta * beta,
                    (4.059078) + (2.597379e-1) * beta + (-2.169951e-2) * beta * beta,
                    (2.086906) + (2.920310e-1) * beta + (-2.560437e-2) * beta * beta,
                ]
            ),
            (2, 3): np.array(
                [
                    (8.073459e-1) + (-2.581232e-2) * beta,
                    (-4.663682e-1) + (-1.030271e-3) * beta,
                    (6.256342e-1) + (4.086881e-3) * beta,
                    (-1.063437) + (-1.235489e-2) * beta,
                    (-4.365989) + (4.391454e-1) * beta + (-1.314615e-2) * beta * beta,
                    (3.854346) + (3.219224e-1) * beta + (-2.587493e-2) * beta * beta,
                    (2.146207) + (3.325620e-1) * beta + (-2.686959e-2) * beta * beta,
                ]
            ),
            (2, 4): np.array(
                [
                    (7.324117e-1) + (-2.727580e-2) * beta,
                    (-4.625614e-1) + (-1.224292e-3) * beta,
                    (3.315871e-1) + (7.216776e-3) * beta,
                    (-1.055706) + (-8.585500e-3) * beta,
                    (-4.571022) + (4.373660e-1) * beta + (-1.221457e-2) * beta * beta,
                    (3.686006) + (3.854493e-1) * beta + (-2.937568e-2) * beta * beta,
                    (2.217893) + (3.641196e-1) * beta + (-2.763824e-2) * beta * beta,
                ]
            ),
            (3, 3): np.array(
                [
                    (8.402943e-1) + (-2.851694e-2) * beta,
                    (-4.727437e-1) + (-1.328784e-3) * beta,
                    (4.724228e-1) + (7.706027e-3) * beta,
                    (-1.213660) + (-3.456656e-2) * beta,
                    (-4.655574) + (4.467685e-1) * beta + (-1.237864e-2) * beta * beta,
                    (3.817178) + (3.503180e-1) * beta + (-2.806506e-2) * beta * beta,
                    (2.313186) + (3.889828e-1) * beta + (-3.120619e-2) * beta * beta,
                ]
            ),
            (4, 4): np.array(
                [
                    (8.088842e-1) + (-2.592379e-2) * beta,
                    (-4.659483e-1) + (-1.041599e-3) * beta,
                    (6.092981e-1) + (1.428402e-3) * beta,
                    (-1.113323) + (-1.031574e-2) * beta,
                    (-4.349145) + (4.236246e-1) * beta + (-1.210668e-2) * beta * beta,
                    (3.828467) + (3.573461e-1) * beta + (-2.759622e-2) * beta * beta,
                    (2.138075) + (3.388072e-1) * beta + (-2.669344e-2) * beta * beta,
                ]
            ),
        }
        key = (l, r)
        if key not in coeffs:
            raise ModelParameterException("ESA_cn omega integral not available for this (l,r)")
        return self._esa_eval(coeffs[key], x)

    def omega_integral_ESA_cn_corr(self, T: float, d1: float, d2: float, d3: float, beta: float, r_e: float) -> float:
        x = math.log(T)
        x0 = 0.7564 * pow(beta, 0.064605)
        sigma_sqr = x0 * x0 * r_e * r_e / 1e20
        return (d1 + d2 * x + d3 * x * x) / sigma_sqr

    def omega_integral_ESA_cc(self, T: float, l: int, r: int, charge1: int, charge2: int, debye_length: float) -> float:
        nl = 1.0 / (1 - (1 + pow(-0.1, l)) / (2 * l + 2))
        Tstar = abs(debye_length * 4 * constants.K_CONST_PI * constants.K_CONST_E0 * constants.K_CONST_K * T / (constants.K_CONST_ELEMENTARY_CHARGE * constants.K_CONST_ELEMENTARY_CHARGE * charge1 * charge2))
        return (
            l
            * nl
            * math.log((4 * Tstar / (constants.K_CONST_EULER * constants.K_CONST_EULER)) * math.exp(self.cc_ar[r - 1] - self.cc_cl[l - 1]) + 1)
            / (Tstar * Tstar * r * (r + 1))
        )

    def omega_integral_ESA_ne(
        self,
        T: float,
        g1: float,
        g2: float,
        g3: float,
        g4: float,
        g5: float,
        g6: float,
        g7: float,
        g8: float,
        g9: float,
        g10: float,
        diameter: float,
    ) -> float:
        x = math.log(T)
        exp_1 = math.exp(x - g1 / g2)
        return (g3 * pow(x, g6) * exp_1 / (exp_1 + 1.0 / exp_1) + g7 * math.exp(-((x - g8) * (x - g8)) / (g9 * g9)) + g4 + g10 * pow(x, g5)) / (
            diameter * diameter
        )

    # ---------------- Backward/forward reaction rate helpers ---------------- #
    @staticmethod
    def p_k_bf_VT(
        T: float,
        vibr_energy_before: float,
        vibr_energy_after: float,
        rot_energy_before: np.ndarray,
        num_rot_levels_before: int,
        rot_energy_after: np.ndarray,
        num_rot_levels_after: int,
        rot_symmetry: int,
        is_rigid_rotator: bool,
    ) -> float:
        if is_rigid_rotator:
            return math.exp((vibr_energy_after - vibr_energy_before) / (constants.K_CONST_K * T))
        return math.exp((vibr_energy_after - vibr_energy_before) / (constants.K_CONST_K * T)) * Approximation.p_Z_rot(T, rot_energy_before, num_rot_levels_before, rot_symmetry) / Approximation.p_Z_rot(
            T, rot_energy_after, num_rot_levels_after, rot_symmetry
        )

    @staticmethod
    def p_k_bf_VV(
        T: float,
        vibr_energy1_before: float,
        vibr_energy1_after: float,
        vibr_energy2_before: float,
        vibr_energy2_after: float,
        rot_energy1_before: np.ndarray,
        num_rot_levels1_before: int,
        rot_energy1_after: np.ndarray,
        num_rot_levels1_after: int,
        rot_symmetry1: int,
        rot_energy2_before: np.ndarray,
        num_rot_levels2_before: int,
        rot_energy2_after: np.ndarray,
        num_rot_levels2_after: int,
        rot_symmetry2: int,
    ) -> float:
        return math.exp((vibr_energy1_after - vibr_energy1_before + vibr_energy2_after - vibr_energy2_before) / (constants.K_CONST_K * T)) * (
            Approximation.p_Z_rot(T, rot_energy1_before, num_rot_levels1_before, rot_symmetry1)
            * Approximation.p_Z_rot(T, rot_energy2_before, num_rot_levels2_before, rot_symmetry2)
        ) / (
            Approximation.p_Z_rot(T, rot_energy1_after, num_rot_levels1_after, rot_symmetry1)
            * Approximation.p_Z_rot(T, rot_energy2_after, num_rot_levels2_after, rot_symmetry2)
        )

    @staticmethod
    def p_k_bf_exch(
        T: float,
        molecule_before_mass: float,
        atom_before_mass: float,
        molecule_after_mass: float,
        atom_after_mass: float,
        diss_energy_before: float,
        diss_energy_after: float,
        vibr_energy_before: float,
        vibr_energy_after: float,
        rot_energy_before: np.ndarray,
        num_rot_levels_before: int,
        rot_symmetry_before: int,
        rot_energy_after: np.ndarray,
        num_rot_levels_after: int,
        rot_symmetry_after: int,
    ) -> float:
        return (
            pow(molecule_before_mass * atom_before_mass / (molecule_after_mass * atom_after_mass), 1.5)
            * (Approximation.p_Z_rot(T, rot_energy_before, num_rot_levels_before, rot_symmetry_before) / Approximation.p_Z_rot(T, rot_energy_after, num_rot_levels_after, rot_symmetry_after))
            * math.exp((vibr_energy_after - vibr_energy_before) / (constants.K_CONST_K * T))
            * math.exp((diss_energy_before - diss_energy_after) / (constants.K_CONST_K * T))
        )

    @staticmethod
    def p_k_bf_diss(
        T: float,
        molecule_mass: float,
        atom1_mass: float,
        atom2_mass: float,
        diss_energy: float,
        vibr_energy: float,
        rot_energy: np.ndarray,
        num_rot_levels: int,
        rot_symmetry: int,
    ) -> float:
        return (
            pow(molecule_mass / (atom1_mass * atom2_mass), 1.5)
            * constants.K_CONST_H
            * constants.K_CONST_H
            * constants.K_CONST_H
            * pow(2 * constants.K_CONST_PI * constants.K_CONST_K * T, -1.5)
            * Approximation.p_Z_rot(T, rot_energy, num_rot_levels, rot_symmetry)
            * math.exp((diss_energy - vibr_energy) / (constants.K_CONST_K * T))
        )

    # Backward/forward rate helpers (public wrappers) ---------------- #
    def k_bf_exch(
        self,
        T: float,
        molecule_before: Molecule,
        atom_before: Particle,
        molecule_after: Molecule,
        atom_after: Particle,
        i: int,
        e: int,
    ) -> float:
        return self.p_k_bf_exch(
            T,
            molecule_before.mass,
            atom_before.mass,
            molecule_after.mass,
            atom_after.mass,
            molecule_before.diss_energy[e],
            molecule_after.diss_energy[e] if molecule_after.diss_energy.size else molecule_before.diss_energy[e],
            molecule_before.vibr_energy[e][i],
            molecule_after.vibr_energy[e][i] if molecule_after.vibr_energy else 0.0,
            molecule_before.rot_energy[e][i],
            molecule_before.num_rot_levels[e][i],
            molecule_before.rot_symmetry,
            molecule_after.rot_energy[e][i] if molecule_after.rot_energy else np.zeros(1),
            molecule_after.num_rot_levels[e][i] if molecule_after.num_rot_levels else 0,
            molecule_after.rot_symmetry if hasattr(molecule_after, "rot_symmetry") else 1,
        )

    def k_bf_diss(self, T: float, molecule: Molecule, atom1: Particle, atom2: Particle, i: int, e: int) -> float:
        return self.p_k_bf_diss(
            T,
            molecule.mass,
            atom1.mass,
            atom2.mass,
            molecule.diss_energy[e],
            molecule.vibr_energy[e][i],
            molecule.rot_energy[e][i],
            molecule.num_rot_levels[e][i],
            molecule.rot_symmetry,
        )

    def k_bf_diss_sym(self, T: float, molecule: Molecule, atom: Particle, i: int, e: int) -> float:
        # symmetric recombination using same atom mass twice
        return self.k_bf_diss(T, molecule, atom, atom, i, e)

    # ---------------- VT integral wrappers ---------------- #
    def integral_VV(self, T: float, degree: int, molecule1: Molecule, molecule2: Molecule, interaction: Interaction, i: int, k: int, delta_i: int, e1: int, e2: int, model) -> float:
        omega_val = abs(molecule1.vibr_energy[e1][i] - molecule1.vibr_energy[e1][i + delta_i]) / (constants.K_CONST_HBAR * abs(delta_i))
        if model == ModelsOmega.RS:
            res = self.integral_VT_FHO_RS(
                T,
                degree,
                interaction.collision_mass,
                interaction.collision_diameter,
                molecule1.reduced_osc_mass,
                molecule1.vibr_energy[e1][i] - molecule1.vibr_energy[e1][i + delta_i],
                i,
                delta_i,
                omega_val,
                molecule1.mA_mAB,
                interaction["alpha_FHO"],
                interaction["E_FHO"],
                interaction["SVT_FHO"],
            )
            if molecule1.rot_symmetry == 1:
                res = 0.5 * (
                    res
                    + self.integral_VT_FHO_RS(
                        T,
                        degree,
                        interaction.collision_mass,
                        interaction.collision_diameter,
                        molecule1.reduced_osc_mass,
                        molecule1.vibr_energy[e1][i] - molecule1.vibr_energy[e1][i + delta_i],
                        i,
                        delta_i,
                        omega_val,
                        molecule1.mB_mAB,
                        interaction["alpha_FHO"],
                        interaction["E_FHO"],
                        interaction["SVT_FHO"],
                    )
                )
            return res
        if model == ModelsOmega.VSS:
            if not interaction.vss_data:
                raise ModelParameterException("VSS data missing for interaction")
            res = self.integral_VT_FHO_VSS(
                T,
                degree,
                interaction.collision_mass,
                interaction.vss_c_cs,
                interaction.vss_omega,
                molecule1.reduced_osc_mass,
                molecule1.vibr_energy[e1][i] - molecule1.vibr_energy[e1][i + delta_i],
                i,
                delta_i,
                omega_val,
                molecule1.mA_mAB,
                interaction["alpha_FHO"],
                interaction["E_FHO"],
                interaction["SVT_FHO"],
            )
            if molecule1.rot_symmetry == 1:
                res = 0.5 * (
                    res
                    + self.integral_VT_FHO_VSS(
                        T,
                        degree,
                        interaction.collision_mass,
                        interaction.vss_c_cs,
                        interaction.vss_omega,
                        molecule1.reduced_osc_mass,
                        molecule1.vibr_energy[e1][i] - molecule1.vibr_energy[e1][i + delta_i],
                        i,
                        delta_i,
                        omega_val,
                        molecule1.mB_mAB,
                        interaction["alpha_FHO"],
                        interaction["E_FHO"],
                        interaction["SVT_FHO"],
                    )
                )
            return res
        raise ModelParameterException(f"VV integral model {model} not supported")

    def integral_VT_FHO_RS(
        self,
        T: float,
        degree: int,
        coll_mass: float,
        diameter: float,
        osc_mass: float,
        delta_E_vibr: float,
        i: int,
        delta_i: int,
        omega: float,
        ram: float,
        alpha_FHO: float,
        E_FHO: float,
        svt_FHO: float,
    ) -> float:
        conversion = math.sqrt(2 * constants.K_CONST_K * T / coll_mass)
        def integrand(g: float) -> float:
            rel_vel = conversion * g
            return (
                pow(g, 2 * degree + 3)
                * self.crosssection_VT_FHO_RS(rel_vel, coll_mass, diameter, osc_mass, delta_E_vibr, i, delta_i, omega, ram, alpha_FHO, E_FHO, svt_FHO)
                * math.exp(-g * g)
            )
        return math.sqrt(constants.K_CONST_K * T / (2 * constants.K_CONST_PI * coll_mass)) * self.integrate_semi_inf(integrand)

    def integral_VT_FHO_VSS(
        self,
        T: float,
        degree: int,
        coll_mass: float,
        vss_c_cs: float,
        vss_omega: float,
        osc_mass: float,
        delta_E_vibr: float,
        i: int,
        delta_i: int,
        omega: float,
        ram: float,
        alpha_FHO: float,
        E_FHO: float,
        svt_FHO: float,
    ) -> float:
        conversion = math.sqrt(2 * constants.K_CONST_K * T / coll_mass)
        def integrand(g: float) -> float:
            rel_vel = conversion * g
            return (
                pow(g, 2 * degree + 3)
                * self.crosssection_VT_FHO_VSS(rel_vel, coll_mass, vss_c_cs, vss_omega, osc_mass, delta_E_vibr, i, delta_i, omega, ram, alpha_FHO, E_FHO, svt_FHO)
                * math.exp(-g * g)
            )
        return math.sqrt(constants.K_CONST_K * T / (2 * constants.K_CONST_PI * coll_mass)) * self.integrate_semi_inf(integrand)

    def integral_VT(
        self,
        T: float,
        degree: int,
        molecule: Molecule,
        interaction: Interaction,
        i: int,
        delta_i: int,
        e: int,
        model: ModelsCsVT,
    ) -> float:
        if model == ModelsCsVT.RS_FHO:
            omega_val = abs(molecule.vibr_energy[e][i] - molecule.vibr_energy[e][i + delta_i]) / (constants.K_CONST_HBAR * abs(delta_i))
            res = self.integral_VT_FHO_RS(
                T,
                degree,
                interaction.collision_mass,
                interaction.collision_diameter,
                molecule.reduced_osc_mass,
                molecule.vibr_energy[e][i] - molecule.vibr_energy[e][i + delta_i],
                i,
                delta_i,
                omega_val,
                molecule.mA_mAB,
                interaction["alpha_FHO"],
                interaction["E_FHO"],
                interaction["SVT_FHO"],
            )
            if molecule.rot_symmetry == 2:
                return res
            return 0.5 * (
                res
                + self.integral_VT_FHO_RS(
                    T,
                    degree,
                    interaction.collision_mass,
                    interaction.collision_diameter,
                    molecule.reduced_osc_mass,
                    molecule.vibr_energy[e][i] - molecule.vibr_energy[e][i + delta_i],
                    i,
                    delta_i,
                    omega_val,
                    molecule.mB_mAB,
                    interaction["alpha_FHO"],
                    interaction["E_FHO"],
                    interaction["SVT_FHO"],
                )
            )
        if model == ModelsCsVT.VSS_FHO:
            if not interaction.vss_data:
                raise ModelParameterException("No VSS data for interaction")
            omega_val = abs(molecule.vibr_energy[e][i] - molecule.vibr_energy[e][i + delta_i]) / (constants.K_CONST_HBAR * abs(delta_i))
            res = self.integral_VT_FHO_VSS(
                T,
                degree,
                interaction.collision_mass,
                interaction.vss_c_cs,
                interaction.vss_omega,
                molecule.reduced_osc_mass,
                molecule.vibr_energy[e][i] - molecule.vibr_energy[e][i + delta_i],
                i,
                delta_i,
                omega_val,
                molecule.mA_mAB,
                interaction["alpha_FHO"],
                interaction["E_FHO"],
                interaction["SVT_FHO"],
            )
            if molecule.rot_symmetry == 2:
                return res
            return 0.5 * (
                res
                + self.integral_VT_FHO_VSS(
                    T,
                    degree,
                    interaction.collision_mass,
                    interaction.vss_c_cs,
                    interaction.vss_omega,
                    molecule.reduced_osc_mass,
                    molecule.vibr_energy[e][i] - molecule.vibr_energy[e][i + delta_i],
                    i,
                    delta_i,
                    omega_val,
                    molecule.mB_mAB,
                    interaction["alpha_FHO"],
                    interaction["E_FHO"],
                    interaction["SVT_FHO"],
                )
            )
        raise ModelParameterException(f"Unsupported VT integral model: {model}")

    def k_VT(self, T: float, molecule: Molecule, interaction: Interaction, i: int, delta_i: int, e: int, model: ModelsKVT) -> float:
        if delta_i == 0:
            raise ValueError("delta_i cannot be zero")
        if model in (ModelsKVT.RS_FHO, ModelsKVT.VSS_FHO):
            if delta_i < 0:
                res = self.k_VT_FHO_RS(T, interaction, molecule, i, delta_i, e) if model == ModelsKVT.RS_FHO else self.k_VT_FHO_VSS(T, interaction, molecule, i, delta_i, e)
                return res
            forward = self.k_VT(T, molecule, interaction, i + delta_i, -delta_i, e, model)
            kbf = self.p_k_bf_VT(
                T,
                molecule.vibr_energy[e][i + delta_i],
                molecule.vibr_energy[e][i],
                molecule.rot_energy[e][i + delta_i],
                molecule.num_rot_levels[e][i + delta_i],
                molecule.rot_energy[e][i],
                molecule.num_rot_levels[e][i],
                molecule.rot_symmetry,
                molecule.rigid_rotator,
            )
            return forward * kbf
        if model == ModelsKVT.SSH:
            if delta_i not in (-1, 1):
                raise ModelParameterException("Only single-quantum transitions allowed for SSH")
            if delta_i == -1:
                if molecule.anharmonic_spectrum:
                    return self.k_VT_SSH(
                        T,
                        i,
                        interaction.collision_mass,
                        interaction.collision_diameter,
                        molecule.vibr_frequency[0],
                        interaction.epsilon,
                        molecule.internuclear_distance,
                        molecule.vibr_energy[0][i] - molecule.vibr_energy[0][i + delta_i],
                        molecule.vibr_energy[0][1] - molecule.vibr_energy[0][0],
                    )
                return self.k_VT_SSH(
                    T,
                    i,
                    interaction.collision_mass,
                    interaction.collision_diameter,
                    molecule.vibr_frequency[0],
                    interaction.epsilon,
                    molecule.internuclear_distance,
                )
            # delta_i == 1
            balance = self.p_k_bf_VT(
                T,
                molecule.vibr_energy[e][i + delta_i],
                molecule.vibr_energy[e][i],
                molecule.rot_energy[e][i + delta_i],
                molecule.num_rot_levels[e][i + delta_i],
                molecule.rot_energy[e][i],
                molecule.num_rot_levels[e][i],
                molecule.rot_symmetry,
                molecule.rigid_rotator,
            )
            if molecule.anharmonic_spectrum:
                return (
                    self.k_VT_SSH(
                        T,
                        i + delta_i,
                        interaction.collision_mass,
                        interaction.collision_diameter,
                        molecule.vibr_frequency[0],
                        interaction.epsilon,
                        molecule.internuclear_distance,
                        molecule.vibr_energy[0][i + delta_i] - molecule.vibr_energy[0][i],
                        molecule.vibr_energy[0][1] - molecule.vibr_energy[0][0],
                    )
                    * balance
                )
            return (
                self.k_VT_SSH(
                    T,
                    i + delta_i,
                    interaction.collision_mass,
                    interaction.collision_diameter,
                    molecule.vibr_frequency[0],
                    interaction.epsilon,
                    molecule.internuclear_distance,
                )
                * balance
            )
        if model == ModelsKVT.BILLING:
            if e != 0:
                raise ModelParameterException("Billing approximation only valid for ground electronic state")
            if interaction.particle1_name in ("N2",) and interaction.particle2_name in ("N2",):
                if delta_i == -1:
                    return self.k_VT_Billing_N2N2(T, i)
                if delta_i == 1:
                    return self.k_VT_Billing_N2N2(T, i + delta_i) * self.p_k_bf_VT(
                        T,
                        molecule.vibr_energy[e][i + delta_i],
                        molecule.vibr_energy[e][i],
                        molecule.rot_energy[e][i + delta_i],
                        molecule.num_rot_levels[e][i + delta_i],
                        molecule.rot_energy[e][i],
                        molecule.num_rot_levels[e][i],
                        molecule.rot_symmetry,
                        molecule.rigid_rotator,
                    )
                raise ModelParameterException("Billing model supports only single-quantum transitions for N2+N2")
            if {"N2", "N"} == {interaction.particle1_name, interaction.particle2_name}:
                if delta_i < 0:
                    return self.k_VT_Billing_N2N(T, i, delta_i)
                return self.k_VT_Billing_N2N(T, i + delta_i, -delta_i) * self.p_k_bf_VT(
                    T,
                    molecule.vibr_energy[e][i + delta_i],
                    molecule.vibr_energy[e][i],
                    molecule.rot_energy[e][i + delta_i],
                    molecule.num_rot_levels[e][i + delta_i],
                    molecule.rot_energy[e][i],
                    molecule.num_rot_levels[e][i],
                    molecule.rot_symmetry,
                    molecule.rigid_rotator,
                )
            raise ModelParameterException(f"No Billing approximation for {interaction.particle1_name}+{interaction.particle2_name}")
        if model == ModelsKVT.PHYS4ENTRY:
            if e != 0:
                raise ModelParameterException("Phys4Entry approximation only valid for ground electronic state")
            if {"N2", "N"} == {interaction.particle1_name, interaction.particle2_name}:
                ladder = int(self.convert_vibr_ladder_N2(molecule.vibr_energy[e][i]))
                if delta_i < 0:
                    return self.k_VT_N2N_p4e(T, ladder, delta_i)
                return self.k_VT_N2N_p4e(T, ladder, -delta_i) * self.p_k_bf_VT(
                    T,
                    molecule.vibr_energy[e][i + delta_i],
                    molecule.vibr_energy[e][i],
                    molecule.rot_energy[e][i + delta_i],
                    molecule.num_rot_levels[e][i + delta_i],
                    molecule.rot_energy[e][i],
                    molecule.num_rot_levels[e][i],
                    molecule.rot_symmetry,
                    molecule.rigid_rotator,
                )
            if {"O2", "O"} == {interaction.particle1_name, interaction.particle2_name}:
                ladder = int(self.convert_vibr_ladder_O2(molecule.vibr_energy[e][i]))
                if delta_i < 0:
                    return self.k_VT_O2O_p4e(T, ladder, delta_i)
                return self.k_VT_O2O_p4e(T, ladder, -delta_i) * self.p_k_bf_VT(
                    T,
                    molecule.vibr_energy[e][i + delta_i],
                    molecule.vibr_energy[e][i],
                    molecule.rot_energy[e][i + delta_i],
                    molecule.num_rot_levels[e][i + delta_i],
                    molecule.rot_energy[e][i],
                    molecule.num_rot_levels[e][i],
                    molecule.rot_symmetry,
                    molecule.rigid_rotator,
                )
            raise ModelParameterException(f"No Phys4Entry approximation for {interaction.particle1_name}+{interaction.particle2_name}")
        raise ModelParameterException(f"Unsupported k_VT model: {model}")

    def k_VV(
        self,
        T: float,
        molecule1: Molecule,
        molecule2: Molecule,
        interaction: Interaction,
        i: int,
        k: int,
        delta_i: int,
        e1: int,
        e2: int,
        model: ModelsKVV,
    ) -> float:
        if abs(delta_i) != 1:
            raise ModelParameterException("k_VV implemented only for single-quantum exchanges (|delta_i|=1)")
        if model == ModelsKVV.SSH:
            omega_e = molecule1.vibr_frequency[0] if molecule1.vibr_frequency.size else 0.0
            return self.k_VV_SSH(T, i, k, interaction.collision_mass, interaction.collision_diameter, omega_e, interaction.epsilon, getattr(molecule1, "internuclear_distance", 0.0))
        if model == ModelsKVV.BILLING:
            if molecule1.name == molecule2.name == "N2":
                if delta_i == -1:
                    return self.k_VV_Billing_N2N2(T, i, k)
                if delta_i == 1:
                    return self.k_VV_Billing_N2N2(T, i + delta_i, k - delta_i) * self.k_bf_VV(T, molecule1, molecule2, i + delta_i, k - delta_i, -delta_i, e1, e2)
            raise ModelParameterException("Billing VV available only for N2+N2 single-quantum exchange")
        raise ModelParameterException(f"Unsupported k_VV model: {model}")

    def k_VT_FHO_RS(self, T: float, interaction: Interaction, molecule: Molecule, i: int, delta_i: int, e: int) -> float:
        omega_val = abs(molecule.vibr_energy[e][i] - molecule.vibr_energy[e][i + delta_i]) / (constants.K_CONST_HBAR * abs(delta_i))
        res = self.k_VT_FHO_core(
            T,
            interaction.collision_mass,
            interaction.collision_diameter,
            molecule.reduced_osc_mass,
            molecule.vibr_energy[e][i] - molecule.vibr_energy[e][i + delta_i],
            i,
            delta_i,
            omega_val,
            molecule.mA_mAB,
            interaction["alpha_FHO"],
            interaction["E_FHO"],
            interaction["SVT_FHO"],
            rs=True,
        )
        if molecule.rot_symmetry == 2:
            return res
        return 0.5 * (
            res
            + self.k_VT_FHO_core(
                T,
                interaction.collision_mass,
                interaction.collision_diameter,
                molecule.reduced_osc_mass,
                molecule.vibr_energy[e][i] - molecule.vibr_energy[e][i + delta_i],
                i,
                delta_i,
                omega_val,
                molecule.mB_mAB,
                interaction["alpha_FHO"],
                interaction["E_FHO"],
                interaction["SVT_FHO"],
                rs=True,
            )
        )

    def k_VT_FHO_VSS(self, T: float, interaction: Interaction, molecule: Molecule, i: int, delta_i: int, e: int) -> float:
        if not interaction.vss_data:
            raise ValueError("VSS data missing for interaction")
        omega_val = abs(molecule.vibr_energy[e][i] - molecule.vibr_energy[e][i + delta_i]) / (constants.K_CONST_HBAR * abs(delta_i))
        res = self.k_VT_FHO_core(
            T,
            interaction.collision_mass,
            interaction.vss_c_cs,
            molecule.reduced_osc_mass,
            molecule.vibr_energy[e][i] - molecule.vibr_energy[e][i + delta_i],
            i,
            delta_i,
            omega_val,
            molecule.mA_mAB,
            interaction["alpha_FHO"],
            interaction["E_FHO"],
            interaction["SVT_FHO"],
            rs=False,
            vss_omega=interaction.vss_omega,
        )
        if molecule.rot_symmetry == 2:
            return res
        return 0.5 * (
            res
            + self.k_VT_FHO_core(
                T,
                interaction.collision_mass,
                interaction.vss_c_cs,
                molecule.reduced_osc_mass,
                molecule.vibr_energy[e][i] - molecule.vibr_energy[e][i + delta_i],
                i,
                delta_i,
                omega_val,
                molecule.mB_mAB,
                interaction["alpha_FHO"],
                interaction["E_FHO"],
                interaction["SVT_FHO"],
                rs=False,
                vss_omega=interaction.vss_omega,
            )
        )

    def k_VT_FHO_core(
        self,
        T: float,
        coll_mass: float,
        diameter_or_c_cs: float,
        osc_mass: float,
        delta_E_vibr: float,
        i: int,
        delta_i: int,
        omega: float,
        ram: float,
        alpha_FHO: float,
        E_FHO: float,
        svt_FHO: float,
        *,
        rs: bool,
        vss_omega: float | None = None,
    ) -> float:
        if rs:
            return self.k_VT_RS_FHO(T, coll_mass, diameter_or_c_cs, osc_mass, delta_E_vibr, i, delta_i, omega, ram, alpha_FHO, E_FHO, svt_FHO)
        return self.k_VT_VSS_FHO(T, coll_mass, diameter_or_c_cs, vss_omega or 0.0, osc_mass, delta_E_vibr, i, delta_i, omega, ram, alpha_FHO, E_FHO, svt_FHO)

    @staticmethod
    def k_VT_RS_FHO(
        T: float,
        coll_mass: float,
        diameter: float,
        osc_mass: float,
        delta_E_vibr: float,
        i: int,
        delta_i: int,
        omega: float,
        ram: float,
        alpha_FHO: float,
        E_FHO: float,
        svt_FHO: float,
    ) -> float:
        return (
            Approximation.k_VT_FHO_prefactor(T, coll_mass, osc_mass, delta_E_vibr)
            * Approximation.crosssection_VT_FHO_RS(1.0, coll_mass, diameter, osc_mass, delta_E_vibr, i, delta_i, omega, ram, alpha_FHO, E_FHO, svt_FHO)
        )

    @staticmethod
    def k_VT_VSS_FHO(
        T: float,
        coll_mass: float,
        vss_c_cs: float,
        vss_omega: float,
        osc_mass: float,
        delta_E_vibr: float,
        i: int,
        delta_i: int,
        omega: float,
        ram: float,
        alpha_FHO: float,
        E_FHO: float,
        svt_FHO: float,
    ) -> float:
        return (
            Approximation.k_VT_FHO_prefactor(T, coll_mass, osc_mass, delta_E_vibr)
            * Approximation.crosssection_VT_FHO_VSS(1.0, coll_mass, vss_c_cs, vss_omega, osc_mass, delta_E_vibr, i, delta_i, omega, ram, alpha_FHO, E_FHO, svt_FHO)
        )

    @staticmethod
    def k_VT_FHO_prefactor(T: float, coll_mass: float, osc_mass: float, delta_E_vibr: float) -> float:
        return (8 * pow(constants.K_CONST_K * T, 1.5) / (math.sqrt(constants.K_CONST_PI * coll_mass) * pow(constants.K_CONST_K * T + abs(delta_E_vibr), 2))) * math.exp(
            -abs(delta_E_vibr) / (constants.K_CONST_K * T)
        )

    @staticmethod
    def k_exch_WRFP(T: float, vibr_energy: float, activation_energy: float, alpha_exch: float, beta_exch: float, A_exch: float, n_exch: float) -> float:
        if activation_energy > alpha_exch * vibr_energy:
            return A_exch * pow(T, n_exch) * math.exp(-(activation_energy - alpha_exch * vibr_energy) / (constants.K_CONST_K * T * beta_exch))
        return A_exch * pow(T, n_exch)

    # ---------------- SSH-based rates ---------------- #
    @staticmethod
    def P_SSH_VT_10(T: float, coll_mass: float, omega_e: float, epsilon: float, diameter: float, r_e: float) -> float:
        alpha = 17.5 / diameter
        omega = 2 * constants.K_CONST_PI * constants.K_CONST_C * omega_e
        # eq.80b (harmonic oscillator)
        return 4 * constants.K_CONST_K * T / coll_mass * (alpha * alpha) / (omega * omega)

    @staticmethod
    def P_SSH_VV_01(T: float, coll_mass: float, omega_e: float, epsilon: float, osc_mass: float, diameter: float, r_e: float) -> float:
        alpha = 17.5 / diameter
        omega = 2 * constants.K_CONST_PI * constants.K_CONST_C * omega_e
        return pow(0.5, 4) * 4 * constants.K_CONST_K * T / osc_mass * alpha * alpha / (omega * omega)

    def k_VT_SSH(self, T: float, i: int, coll_mass: float, diameter: float, omega_e: float, epsilon: float, r_e: float) -> float:
        return self.p_Z_coll(T, 1.0, coll_mass, diameter) * i * self.P_SSH_VT_10(T, coll_mass, omega_e, epsilon, diameter, r_e)

    def k_VV_SSH(self, T: float, i: int, k: int, coll_mass: float, diameter: float, omega_e: float, epsilon: float, r_e: float) -> float:
        return self.p_Z_coll(T, 1.0, coll_mass, diameter) * (i + 1) * (k + 1) * self.P_SSH_VV_01(T, coll_mass, omega_e, epsilon, coll_mass, diameter, r_e)

    @staticmethod
    def k_VT_SSH_anharm(
        T: float,
        i: int,
        coll_mass: float,
        diameter: float,
        omega_e: float,
        epsilon: float,
        r_e: float,
        Delta_E_vibr: float,
        vibr_energy_1: float,
    ) -> float:
        alpha = 17.5 / diameter
        gamma0 = constants.K_CONST_PI * math.sqrt(coll_mass / (2 * constants.K_CONST_K * T)) / (alpha * constants.K_CONST_HBAR)
        gammai = gamma0 * (vibr_energy_1 - 2 * i * Delta_E_vibr)
        gamma0 *= vibr_energy_1
        delta_VT = Delta_E_vibr / vibr_energy_1
        if gammai >= 20:
            delta_VT *= 4 * pow(gamma0, 2.0 / 3.0)
        else:
            delta_VT *= (4.0 / 3.0) * gamma0
        return Approximation.p_Z_coll(T, 1.0, coll_mass, diameter) * i * Approximation.P_SSH_VT_10(T, coll_mass, omega_e, epsilon, diameter, r_e) * math.exp(i * delta_VT) * math.exp(
            -i * Delta_E_vibr / (constants.K_CONST_K * T)
        )

    # ---------------- VV rate coefficients (Billing) ---------------- #
    def k_VV_Billing_N2N2(self, T: float, i: int, k: int) -> float:
        a = [4.40e-17, 2.71e-13, 1.28e-18, 2.21e-18]
        b = [0.0904, -0.0719, 0.0360, -0.0381]
        c = [20.0, 14.0, 7.0, 10.0]
        idx = i + k
        if idx > 3:
            idx = 3
        return a[idx] * pow(T, b[idx]) * math.exp(-c[idx] * 1.602176565e-19 / (constants.K_CONST_K * T))

    def k_VT_Billing_N2N2(self, T: float, i: int) -> float:
        return 1e-6 * i * math.exp(-3.24093 - 140.69597 * pow(T, -0.2)) * math.exp((0.26679 - 6.99237e-5 * T + 4.70073e-9 * T * T) * (i - 1))

    def k_VT_Billing_N2N(self, T: float, i: int, delta_i: int) -> float:
        return 1e-6 * math.exp(
            -25.708
            - 5633.1543 / T
            + (0.1554 - 111.3426 / T) * delta_i
            + (0.0054 - 2.189 / T) * delta_i * delta_i
            + i
            * (
                0.0536
                + 122.4835 / T
                - (0.0013 - 4.2365 / T) * delta_i
                + (-0.0001197 + 0.0807 / T) * delta_i * delta_i
            )
        )

    # Phys4Entry VT rates (N2/N and O2/O)
    def k_VT_N2N_p4e(self, T: float, i: int, delta_i: int) -> float:
        delta = -delta_i
        a = [0.0] * 5
        if delta == 1:
            for j in range(5):
                a[j] = (
                    self.p4e_n2n_vt_bij1[j][0]
                    + self.p4e_n2n_vt_bij1[j][1] * i
                    + self.p4e_n2n_vt_bij1[j][2] * (i * i)
                    + self.p4e_n2n_vt_bij1[j][3] * (i * i * i)
                    + self.p4e_n2n_vt_bij1[j][4] * math.log(i)
                )
            return math.exp(a[0] + a[1] / T + a[2] / (T * T) + a[3] / (T * T * T) + a[4] * math.log(T))
        if 2 <= delta <= 5:
            for j in range(5):
                a[j] = (
                    (self.p4e_n2n_vt_bij2[j][0] + self.p4e_n2n_vt_cij2[j][0] * delta)
                    + (self.p4e_n2n_vt_bij2[j][1] + self.p4e_n2n_vt_cij2[j][1] * delta) * i
                    + (self.p4e_n2n_vt_bij2[j][2] + self.p4e_n2n_vt_cij2[j][2] * delta) * (i * i)
                    + (self.p4e_n2n_vt_bij2[j][3] + self.p4e_n2n_vt_cij2[j][3] * delta) * (i * i * i)
                    + (self.p4e_n2n_vt_bij2[j][4] + self.p4e_n2n_vt_cij2[j][4] * delta) * math.log(i)
                )
            return 1e-6 * math.exp(a[0] + a[1] / T + a[2] / (T * T) + a[3] / (T * T * T) + a[4] * math.log(T))
        if 6 <= delta <= 10:
            for j in range(5):
                a[j] = (
                    (self.p4e_n2n_vt_bij3[j][0] + self.p4e_n2n_vt_cij3[j][0] * delta)
                    + (self.p4e_n2n_vt_bij3[j][1] + self.p4e_n2n_vt_cij3[j][1] * delta) * i
                    + (self.p4e_n2n_vt_bij3[j][2] + self.p4e_n2n_vt_cij3[j][2] * delta) * (i * i)
                    + (self.p4e_n2n_vt_bij3[j][3] + self.p4e_n2n_vt_cij3[j][3] * delta) * (i * i * i)
                    + (self.p4e_n2n_vt_bij3[j][4] + self.p4e_n2n_vt_cij3[j][4] * delta) * math.log(i)
                )
            return 1e-6 * math.exp(a[0] + a[1] / T + a[2] / (T * T) + a[3] / (T * T * T) + a[4] * math.log(T))
        if 11 <= delta <= 20:
            for j in range(5):
                a[j] = (
                    (self.p4e_n2n_vt_bij4[j][0] + self.p4e_n2n_vt_cij4[j][0] * delta)
                    + (self.p4e_n2n_vt_bij4[j][1] + self.p4e_n2n_vt_cij4[j][1] * delta) * i
                    + (self.p4e_n2n_vt_bij4[j][2] + self.p4e_n2n_vt_cij4[j][2] * delta) * (i * i)
                    + (self.p4e_n2n_vt_bij4[j][3] + self.p4e_n2n_vt_cij4[j][3] * delta) * (i * i * i)
                    + (self.p4e_n2n_vt_bij4[j][4] + self.p4e_n2n_vt_cij4[j][4] * delta) * math.log(i)
                )
            return 1e-6 * math.exp(a[0] + a[1] / T + a[2] / (T * T) + a[3] / (T * T * T) + a[4] * math.log(T))
        if 21 <= delta <= 30:
            for j in range(5):
                a[j] = (
                    (self.p4e_n2n_vt_bij5[j][0] + self.p4e_n2n_vt_cij5[j][0] * delta)
                    + (self.p4e_n2n_vt_bij5[j][1] + self.p4e_n2n_vt_cij5[j][1] * delta) * i
                    + (self.p4e_n2n_vt_bij5[j][2] + self.p4e_n2n_vt_cij5[j][2] * delta) * (i * i)
                    + (self.p4e_n2n_vt_bij5[j][3] + self.p4e_n2n_vt_cij5[j][3] * delta) * (i * i * i)
                    + (self.p4e_n2n_vt_bij5[j][4] + self.p4e_n2n_vt_cij5[j][4] * delta) * math.log(i)
                )
            return 1e-6 * math.exp(a[0] + a[1] / T + a[2] / (T * T) + a[3] / (T * T * T) + a[4] * math.log(T))
        # delta >=31
        a = [0.0] * 4
        for j in range(4):
            a[j] = (
                (self.p4e_n2n_vt_bij6[j][0] + self.p4e_n2n_vt_cij6[j][0] * delta)
                + (self.p4e_n2n_vt_bij6[j][1] + self.p4e_n2n_vt_cij6[j][0] * delta) * i
                + (self.p4e_n2n_vt_bij6[j][2] + self.p4e_n2n_vt_cij6[j][0] * delta) * (i * i)
            )
        return 1e-6 * math.exp(a[0] + a[1] / T + a[2] / (T**4) + a[3] / math.log(T))

    def k_VT_O2O_p4e(self, T: float, k: int, delta_i: int) -> float:
        delta = -delta_i
        if delta < 1:
            raise ValueError("delta_i must be negative for de-excitation")
        Bij = np.zeros((3, 5))
        if delta == 1:
            coeff = self.p4e_o2o_vt_Cijk1
        elif 1 < delta <= 10:
            coeff = self.p4e_o2o_vt_Cijk2
        elif 10 < delta <= 20:
            coeff = self.p4e_o2o_vt_Cijk3
        elif 20 < delta <= 30:
            coeff = self.p4e_o2o_vt_Cijk4
        else:
            return -1.0
        for i_idx in range(3):
            for j_idx in range(5):
                Bij[i_idx][j_idx] = (
                    coeff[i_idx][j_idx][0]
                    + coeff[i_idx][j_idx][1] * math.log(delta)
                    + coeff[i_idx][j_idx][2] * delta * math.exp(-delta)
                    + coeff[i_idx][j_idx][3] * delta * delta
                )
        Ai = np.zeros(3)
        for i_idx in range(3):
            Ai[i_idx] = Bij[i_idx][0] + Bij[i_idx][1] * math.log(k) + (Bij[i_idx][2] + Bij[i_idx][3] * k + Bij[i_idx][4] * k * k) / (1e21 + math.exp(k))
        return 1e-6 * (1.0 / 27.0 + math.exp(Ai[0] + Ai[1] / math.log(T) + Ai[2] * math.log(T)))

    # ---------------- Probability wrappers for VV/VT/dissociation ---------------- #
    def probability_VV(
        self,
        rel_vel: float,
        molecule1: Molecule,
        molecule2: Molecule,
        interaction: Interaction,
        i: int,
        k: int,
        delta_i: int,
        e1: int,
        e2: int,
        model: ModelsProbVV = ModelsProbVV.FHO,
    ) -> float:
        if model != ModelsProbVV.FHO:
            raise ModelParameterException(f"Unsupported VV probability model: {model}")
        abs_delta = abs(delta_i)
        omega1 = abs(molecule1.vibr_energy[e1][i] - molecule1.vibr_energy[e1][i + delta_i]) / abs_delta
        omega2 = abs(molecule2.vibr_energy[e2][k] - molecule2.vibr_energy[e2][k - delta_i]) / abs_delta
        return self.probability_VV_FHO(
            rel_vel,
            interaction.collision_mass,
            molecule1.vibr_energy[e1][i] - molecule1.vibr_energy[e1][i + delta_i] + molecule2.vibr_energy[e2][k] - molecule2.vibr_energy[e2][k - delta_i],
            i,
            k,
            delta_i,
            omega1,
            omega2,
            abs(omega1 - omega2),
            interaction["alpha_FHO"],
        )

    def probability_VT(self, rel_vel: float, molecule: Molecule, interaction: Interaction, i: int, delta_i: int, e: int, model: ModelsProbVT = ModelsProbVT.FHO) -> float:
        if model != ModelsProbVT.FHO:
            raise ModelParameterException(f"Unsupported VT probability model: {model}")
        delta_E = molecule.vibr_energy[e][i] - molecule.vibr_energy[e][i + delta_i]
        omega = abs(delta_E) / (constants.K_CONST_HBAR * abs(delta_i))
        res = self.probability_VT_FHO(
            rel_vel,
            interaction.collision_mass,
            molecule.reduced_osc_mass,
            delta_E,
            i,
            delta_i,
            omega,
            molecule.mA_mAB,
            interaction["alpha_FHO"],
            interaction["E_FHO"],
            interaction["SVT_FHO"],
        )
        if molecule.rot_symmetry == 1:
            res = 0.5 * (
                res
                + self.probability_VT_FHO(
                    rel_vel,
                    interaction.collision_mass,
                    molecule.reduced_osc_mass,
                    delta_E,
                    i,
                    delta_i,
                    omega,
                    molecule.mB_mAB,
                    interaction["alpha_FHO"],
                    interaction["E_FHO"],
                    interaction["SVT_FHO"],
                )
            )
        return res

    def k_bf_VT(self, T: float, molecule: Molecule, i: int, delta_i: int, e: int) -> float:
        return self.p_k_bf_VT(
            T,
            molecule.vibr_energy[e][i],
            molecule.vibr_energy[e][i + delta_i],
            molecule.rot_energy[e][i],
            molecule.num_rot_levels[e][i],
            molecule.rot_energy[e][i + delta_i],
            molecule.num_rot_levels[e][i + delta_i],
            molecule.rot_symmetry,
            molecule.rigid_rotator,
        )

    def k_bf_VV(self, T: float, molecule1: Molecule, molecule2: Molecule, i: int, k: int, delta_i: int, e1: int, e2: int) -> float:
        return self.p_k_bf_VV(
            T,
            molecule1.vibr_energy[e1][i],
            molecule1.vibr_energy[e1][i + delta_i],
            molecule2.vibr_energy[e2][k],
            molecule2.vibr_energy[e2][k - delta_i],
            molecule1.rot_energy[e1][i],
            molecule1.num_rot_levels[e1][i],
            molecule1.rot_energy[e1][i + delta_i],
            molecule1.num_rot_levels[e1][i + delta_i],
            molecule1.rot_symmetry,
            molecule2.rot_energy[e2][k],
            molecule2.num_rot_levels[e2][k],
            molecule2.rot_energy[e2][k - delta_i],
            molecule2.num_rot_levels[e2][k - delta_i],
            molecule2.rot_symmetry,
        )

    def probability_diss(self, rel_vel: float, molecule: Molecule, interaction: Interaction, i: int, e: int, model) -> float:
        if model == ModelsProbDiss.THRESH_CMASS_VIBR:
            return self.p_probability_diss(rel_vel, interaction.collision_mass, molecule.diss_energy[e], molecule.vibr_energy[e][i], True)
        if model == ModelsProbDiss.THRESH_VIBR:
            return self.p_probability_diss(rel_vel, interaction.collision_mass, molecule.diss_energy[e], molecule.vibr_energy[e][i], False)
        if model == ModelsProbDiss.THRESH_CMASS:
            return self.p_probability_diss(rel_vel, interaction.collision_mass, molecule.diss_energy[e], 0.0, True)
        if model == ModelsProbDiss.THRESH:
            return self.p_probability_diss(rel_vel, interaction.collision_mass, molecule.diss_energy[e], 0.0, False)
        raise ModelParameterException(f"Unknown dissociation probability model: {model}")

    # ---------------- Cross sections (high-level dispatch) ---------------- #
    def crosssection_elastic(self, rel_vel: float, interaction: Interaction, model: ModelsCsElastic) -> float:
        if model == ModelsCsElastic.VSS:
            if not interaction.vss_data:
                raise ModelParameterException(f"No VSS data for {interaction.particle1_name}+{interaction.particle2_name}")
            return self.crosssection_elastic_VSS_cs(rel_vel, interaction.vss_c_cs, interaction.vss_omega)
        if model == ModelsCsElastic.RS:
            return self.crosssection_elastic_RS(interaction.collision_diameter)
        raise ModelParameterException(f"Unsupported elastic cross section model: {model}")

    def crosssection_VT(self, rel_vel: float, molecule: Molecule, interaction: Interaction, i: int, delta_i: int, e: int, model: ModelsCsVT) -> float:
        omega_val = abs(molecule.vibr_energy[e][i] - molecule.vibr_energy[e][i + delta_i]) / (constants.K_CONST_HBAR * abs(delta_i))
        if model == ModelsCsVT.RS_FHO:
            res = self.crosssection_VT_FHO_RS(
                rel_vel,
                interaction.collision_mass,
                interaction.collision_diameter,
                molecule.reduced_osc_mass,
                molecule.vibr_energy[e][i] - molecule.vibr_energy[e][i + delta_i],
                i,
                delta_i,
                omega_val,
                molecule.mA_mAB,
                interaction["alpha_FHO"],
                interaction["E_FHO"],
                interaction["SVT_FHO"],
            )
            if molecule.rot_symmetry == 1:
                res = 0.5 * (
                    res
                    + self.crosssection_VT_FHO_RS(
                        rel_vel,
                        interaction.collision_mass,
                        interaction.collision_diameter,
                        molecule.reduced_osc_mass,
                        molecule.vibr_energy[e][i] - molecule.vibr_energy[e][i + delta_i],
                        i,
                        delta_i,
                        omega_val,
                        molecule.mB_mAB,
                        interaction["alpha_FHO"],
                        interaction["E_FHO"],
                        interaction["SVT_FHO"],
                    )
                )
            return res
        if model == ModelsCsVT.VSS_FHO:
            if not interaction.vss_data:
                raise ModelParameterException("VSS data missing for interaction")
            res = self.crosssection_VT_FHO_VSS(
                rel_vel,
                interaction.collision_mass,
                interaction.vss_c_cs,
                interaction.vss_omega,
                molecule.reduced_osc_mass,
                molecule.vibr_energy[e][i] - molecule.vibr_energy[e][i + delta_i],
                i,
                delta_i,
                omega_val,
                molecule.mA_mAB,
                interaction["alpha_FHO"],
                interaction["E_FHO"],
                interaction["SVT_FHO"],
            )
            if molecule.rot_symmetry == 1:
                res = 0.5 * (
                    res
                    + self.crosssection_VT_FHO_VSS(
                        rel_vel,
                        interaction.collision_mass,
                        interaction.vss_c_cs,
                        interaction.vss_omega,
                        molecule.reduced_osc_mass,
                        molecule.vibr_energy[e][i] - molecule.vibr_energy[e][i + delta_i],
                        i,
                        delta_i,
                        omega_val,
                        molecule.mB_mAB,
                        interaction["alpha_FHO"],
                        interaction["E_FHO"],
                        interaction["SVT_FHO"],
                    )
                )
            return res
        raise ModelParameterException(f"Unsupported VT cross section model: {model}")

    def crosssection_diss(self, rel_vel: float, molecule: Molecule, interaction: Interaction, i: int, e: int, model: ModelsCsDiss) -> float:
        # Mirror the exact call patterns used in the C++ switch-case (including defaults/order).
        if model == ModelsCsDiss.RS_THRESH_CMASS_VIBR:
            return self.crosssection_diss_RS(rel_vel, interaction.collision_mass, interaction.collision_diameter, molecule.diss_energy[e], molecule.vibr_energy[e][i], True)
        if model == ModelsCsDiss.RS_THRESH_VIBR:
            return self.crosssection_diss_RS(rel_vel, interaction.collision_mass, interaction.collision_diameter, molecule.diss_energy[e], molecule.vibr_energy[e][i], False)
        if model == ModelsCsDiss.RS_THRESH_CMASS:
            # C++ passes 'true' as the vibrational-energy argument and relies on the default center_of_mass=True.
            return self.crosssection_diss_RS(rel_vel, interaction.collision_mass, interaction.collision_diameter, molecule.diss_energy[e], 1.0, True)
        if model == ModelsCsDiss.RS_THRESH:
            # C++ passes 'false' as the vibrational-energy argument, center_of_mass defaults to True.
            return self.crosssection_diss_RS(rel_vel, interaction.collision_mass, interaction.collision_diameter, molecule.diss_energy[e], 0.0, True)
        if model == ModelsCsDiss.VSS_THRESH_CMASS_VIBR:
            if not interaction.vss_data:
                raise ModelParameterException("No VSS data for interaction")
            return self.crosssection_diss_VSS(rel_vel, interaction.collision_mass, interaction.vss_c_cs, interaction.vss_omega, molecule.diss_energy[e], molecule.vibr_energy[e][i], True)
        if model == ModelsCsDiss.VSS_THRESH_VIBR:
            if not interaction.vss_data:
                raise ModelParameterException("No VSS data for interaction")
            return self.crosssection_diss_VSS(rel_vel, interaction.collision_mass, interaction.vss_c_cs, interaction.vss_omega, molecule.diss_energy[e], molecule.vibr_energy[e][i], False)
        if model == ModelsCsDiss.VSS_THRESH_CMASS:
            if not interaction.vss_data:
                raise ModelParameterException("No VSS data for interaction")
            return self.crosssection_diss_VSS(rel_vel, interaction.collision_mass, interaction.vss_c_cs, interaction.vss_omega, molecule.diss_energy[e], 1.0, True)
        if model == ModelsCsDiss.VSS_THRESH:
            if not interaction.vss_data:
                raise ModelParameterException("No VSS data for interaction")
            return self.crosssection_diss_VSS(rel_vel, interaction.collision_mass, interaction.vss_c_cs, interaction.vss_omega, molecule.diss_energy[e], 0.0, True)
        if model == ModelsCsDiss.ILT:
            if e == 0:
                if molecule.name == "N2" and ("N" in (interaction.particle1_name, interaction.particle2_name)):
                    return self.crosssection_diss_ILT_N2N(rel_vel, interaction.collision_mass, molecule.vibr_energy[e][i], self.convert_vibr_ladder_N2(molecule.vibr_energy[e][i]))
                if molecule.name == "O2" and ("O" in (interaction.particle1_name, interaction.particle2_name)):
                    return self.crosssection_diss_ILT_O2O(rel_vel, interaction.collision_mass, molecule.vibr_energy[e][i], self.convert_vibr_ladder_O2(molecule.vibr_energy[e][i]))
        raise ModelParameterException(f"Dissociation cross section model not available: {model}")

    def crosssection_diss_ILT_N2N(self, rel_vel: float, coll_mass: float, vibr_energy: float, i: float) -> float:
        t = coll_mass * rel_vel * rel_vel / (2.0 * constants.K_CONST_K)
        if i <= 8:
            c1 = 1.786e-18
        elif i <= 34:
            c1 = 1.71e-18
        elif i <= 52:
            c1 = 1.68e-18
        else:
            c1 = 1.66e-18
        if t < (c1 - vibr_energy) / constants.K_CONST_K:
            return 0.0
        return (
            math.sqrt(constants.K_CONST_PI * coll_mass / (8 * constants.K_CONST_K))
            * 7.16e-2
            * (5.24e-19 * i * i * i - 7.41e-17 * i * i + 6.42e-15 * i + 7.3e-14)
            * pow(t - (c1 - vibr_energy) / constants.K_CONST_K, 0.25)
            / (0.91 * t)
        )

    def crosssection_diss_ILT_O2O(self, rel_vel: float, coll_mass: float, vibr_energy: float, i: float) -> float:
        t = coll_mass * rel_vel * rel_vel / (2 * constants.K_CONST_K)
        c1 = 0.3867 * i * i * i - 2.7425 * i * i - 1901.9 * i + 61696
        if t < c1:
            return 0.0
        if i <= 31:
            c2 = 1.63e-9 * i * i * i - 1.25e-7 * i * i + 3.24e-6 * i + 7.09e-5
        elif i <= 37:
            c2 = -6.67e-6 * i * i + 4.65e-4 * i - 7.91e-3
        else:
            c2 = 7.83e-7 * i * i * i * i - 1.31e-4 * i * i * i + 8.24e-3 * i * i - 0.23 * i + 2.4049
        return math.sqrt(constants.K_CONST_PI * coll_mass / (8 * constants.K_CONST_K)) * 1.53e-10 * c2 * pow(t - c1, 0.4) / (0.89 * t)

    # ---------------- Arrhenius / exchange / dissociation rate coefficients ---------------- #
    @staticmethod
    def k_Arrhenius(T: float, arrhenius_A: float, arrhenius_n: float, energy: float) -> float:
        return arrhenius_A * pow(T, arrhenius_n) * math.exp(-energy / (constants.K_CONST_K * T))

    def k_exch(
        self,
        T: float,
        molecule: Molecule,
        atom,
        interaction: Interaction,
        i: int,
        e: int,
        num_electron_levels: int,
        model: ModelsKExch,
        molecule_prod: Molecule | None = None,
        k_prod: int = 0,
    ) -> float:
        if num_electron_levels == -1:
            num_electron_levels = molecule.num_electron_levels
        if model == ModelsKExch.ARRH_SCANLON:
            if (molecule.name, atom.name) in (("N2", "O"), ("O2", "N"), ("NO", "O"), ("NO", "N")) or (atom.name, molecule.name) in (("N2", "O"), ("O2", "N"), ("NO", "O"), ("NO", "N")):
                return self.k_Arrhenius(T, interaction["exch,Arrh_A,Scanlon"], interaction["exch,Arrh_n,Scanlon"], interaction["exch,Ea,Scanlon"])
            raise ValueError(f"No Scanlon exchange data for {molecule.name}+{atom.name}")
        if model == ModelsKExch.ARRH_PARK:
            if (molecule.name, atom.name) in (("N2", "O"), ("NO", "O")) or (atom.name, molecule.name) in (("N2", "O"), ("NO", "O")):
                return self.k_Arrhenius(T, interaction["exch,Arrh_A,Park"], interaction["exch,Arrh_n,Park"], interaction["exch,Ed,Park"])
            raise ValueError(f"No Park exchange data for {molecule.name}+{atom.name}")
        if model == ModelsKExch.WARNATZ:
            if (molecule.name, atom.name) in (("N2", "O"), ("O2", "N")) or (atom.name, molecule.name) in (("N2", "O"), ("O2", "N")):
                return self.k_exch_WRFP(T, molecule.vibr_energy[e][i], interaction["exch,Ea,WRFP"], 1.0, 1.0, interaction["exch,A,Warnatz"] * (i + 1), interaction["exch,n,Warnatz"])
            raise ValueError(f"No Warnatz exchange data for {molecule.name}+{atom.name}")
        if model == ModelsKExch.RF:
            if (molecule.name, atom.name) in (("N2", "O"), ("O2", "N")) or (atom.name, molecule.name) in (("N2", "O"), ("O2", "N")):
                return self.k_exch_WRFP(T, molecule.vibr_energy[e][i], interaction["exch,Ea,WRFP"], interaction["exch,alpha,RF"], 1.0, interaction["exch,A,RFP"], interaction["exch,n,RF"])
            raise ValueError(f"No RF exchange data for {molecule.name}+{atom.name}")
        if model == ModelsKExch.POLAK:
            if (molecule.name, atom.name) in (("N2", "O"), ("O2", "N")) or (atom.name, molecule.name) in (("N2", "O"), ("O2", "N")):
                return self.k_exch_WRFP(
                    T,
                    molecule.vibr_energy[e][i],
                    interaction["exch,Ea,WRFP"],
                    interaction["exch,alpha,Polak"],
                    interaction["exch,beta,Polak"],
                    interaction["exch,A,RFP"],
                    interaction["exch,n,Polak"],
                )
            raise ValueError(f"No Polak exchange data for {molecule.name}+{atom.name}")
        if model in (
            ModelsKExch.MALIAT_D6K_ARRH_SCANLON,
            ModelsKExch.MALIAT_3T_ARRH_SCANLON,
            ModelsKExch.MALIAT_INF_ARRH_SCANLON,
            ModelsKExch.MALIAT_D6K_ARRH_PARK,
            ModelsKExch.MALIAT_3T_ARRH_PARK,
            ModelsKExch.MALIAT_INF_ARRH_PARK,
        ):
            # choose Arrhenius source and U parameter
            if model in (
                ModelsKExch.MALIAT_D6K_ARRH_SCANLON,
                ModelsKExch.MALIAT_3T_ARRH_SCANLON,
                ModelsKExch.MALIAT_INF_ARRH_SCANLON,
            ):
                arrh = ("exch,Arrh_A,Scanlon", "exch,Arrh_n,Scanlon", "exch,Ea,Scanlon")
            else:
                arrh = ("exch,Arrh_A,Park", "exch,Arrh_n,Park", "exch,Ed,Park")
            if model in (ModelsKExch.MALIAT_D6K_ARRH_SCANLON, ModelsKExch.MALIAT_D6K_ARRH_PARK):
                U = molecule.diss_energy[e] / (6 * constants.K_CONST_K)
            elif model in (ModelsKExch.MALIAT_3T_ARRH_SCANLON, ModelsKExch.MALIAT_3T_ARRH_PARK):
                U = 3 * T
            else:
                U = None  # infinity
            # species check
            if (molecule.name, atom.name) not in (("N2", "O"), ("O2", "N"), ("NO", "O"), ("NO", "N")) and (atom.name, molecule.name) not in (("N2", "O"), ("O2", "N"), ("NO", "O"), ("NO", "N")):
                raise ValueError(f"No Maliat exchange data for {molecule.name}+{atom.name}")
            k_arrh = self.k_Arrhenius(T, interaction[arrh[0]], interaction[arrh[1]], interaction[arrh[2]])
            if molecule_prod is None:
                raise ValueError("molecule_prod must be provided for Maliat exchange models")
            vibr_energy_product = molecule_prod.vibr_energy[0][k_prod] if molecule_prod.vibr_energy else 0.0
            return k_arrh * self.C_aliat(
                T,
                molecule.electron_energy,
                molecule.statistical_weight,
                num_electron_levels,
                molecule.vibr_energy,
                molecule.num_vibr_levels,
                vibr_energy_product,
                interaction["exch,Ea,Scanlon"],
                i,
                e,
                U,
            )
        raise ModelParameterException(f"Unsupported exchange model: {model}")

    # ---------------- Dissociation rates ---------------- #
    def k_diss_RS(self, T: float, coll_mass: float, diameter: float, diss_energy: float, vibr_energy: float, center_of_mass: bool = True) -> float:
        return 8.0 * self.integral_diss_RS(T, 0, coll_mass, diameter, diss_energy, vibr_energy, center_of_mass)

    def k_diss_VSS(self, T: float, coll_mass: float, vss_c_cs: float, vss_omega: float, diss_energy: float, vibr_energy: float, center_of_mass: bool = True) -> float:
        return 8.0 * self.integral_diss_VSS(T, 0, coll_mass, vss_c_cs, vss_omega, diss_energy, vibr_energy, center_of_mass)

    def k_diss_ILT_N2N(self, T: float, coll_mass: float, vibr_energy: float, i: float) -> float:
        return 8.0 * self.integral_diss_ILT_N2N(T, 0, coll_mass, vibr_energy, i)

    def k_diss_ILT_O2O(self, T: float, coll_mass: float, vibr_energy: float, i: float) -> float:
        return 8.0 * self.integral_diss_ILT_O2O(T, 0, coll_mass, vibr_energy, i)

    def integral_diss_RS(self, T: float, degree: int, coll_mass: float, diameter: float, diss_energy: float, vibr_energy: float, center_of_mass: bool = True) -> float:
        conversion = math.sqrt(2 * constants.K_CONST_K * T / coll_mass)
        def integrand(g: float) -> float:
            rel_vel = conversion * g
            return pow(g, 2 * degree + 3) * self.crosssection_diss_RS(rel_vel, coll_mass, diameter, diss_energy, vibr_energy, center_of_mass) * math.exp(-g * g)
        lower = self.min_vel_diss(coll_mass, diss_energy, vibr_energy) / conversion
        return math.sqrt(constants.K_CONST_K * T / (2 * constants.K_CONST_PI * coll_mass)) * self.integrate_semi_inf(integrand, lower)

    def integral_diss_VSS(self, T: float, degree: int, coll_mass: float, vss_c_cs: float, vss_omega: float, diss_energy: float, vibr_energy: float, center_of_mass: bool = True) -> float:
        conversion = math.sqrt(2 * constants.K_CONST_K * T / coll_mass)
        def integrand(g: float) -> float:
            rel_vel = conversion * g
            return pow(g, 2 * degree + 3) * self.crosssection_diss_VSS(rel_vel, coll_mass, vss_c_cs, vss_omega, diss_energy, vibr_energy, center_of_mass) * math.exp(-g * g)
        lower = self.min_vel_diss(coll_mass, diss_energy, vibr_energy) / conversion
        return math.sqrt(constants.K_CONST_K * T / (2 * constants.K_CONST_PI * coll_mass)) * self.integrate_semi_inf(integrand, lower)

    def integral_diss_ILT_N2N(self, T: float, degree: int, coll_mass: float, vibr_energy: float, i: float) -> float:
        # Approximate from rate fit (k = 8 * integral) => integral = k/8
        conversion = math.sqrt(2 * constants.K_CONST_K * T / coll_mass)
        lower = self.min_vel_diss_ILT_N2N(coll_mass, vibr_energy, i) / conversion
        def integrand(g: float) -> float:
            rel_vel = conversion * g
            return pow(g, 2 * degree + 3) * self.crosssection_diss_ILT_N2N(rel_vel, coll_mass, vibr_energy, i) * math.exp(-g * g)
        return math.sqrt(constants.K_CONST_K * T / (2 * constants.K_CONST_PI * coll_mass)) * self.integrate_semi_inf(integrand, lower)

    def integral_diss_ILT_O2O(self, T: float, degree: int, coll_mass: float, vibr_energy: float, i: float) -> float:
        conversion = math.sqrt(2 * constants.K_CONST_K * T / coll_mass)
        lower = self.min_vel_diss_ILT_O2O(coll_mass, i) / conversion
        def integrand(g: float) -> float:
            rel_vel = conversion * g
            return pow(g, 2 * degree + 3) * self.crosssection_diss_ILT_O2O(rel_vel, coll_mass, vibr_energy, i) * math.exp(-g * g)
        return math.sqrt(constants.K_CONST_K * T / (2 * constants.K_CONST_PI * coll_mass)) * self.integrate_semi_inf(integrand, lower)

    def integral_diss(self, T: float, degree: int, molecule: Molecule, interaction: Interaction, i: int, e: int, model: ModelsCsDiss) -> float:
        if model == ModelsCsDiss.RS_THRESH_CMASS_VIBR:
            return self.integral_diss_RS(T, degree, interaction.collision_mass, interaction.collision_diameter, molecule.diss_energy[e], molecule.vibr_energy[e][i], True)
        if model == ModelsCsDiss.RS_THRESH_VIBR:
            return self.integral_diss_RS(T, degree, interaction.collision_mass, interaction.collision_diameter, molecule.diss_energy[e], molecule.vibr_energy[e][i], False)
        if model == ModelsCsDiss.RS_THRESH_CMASS:
            return self.integral_diss_RS(T, degree, interaction.collision_mass, interaction.collision_diameter, molecule.diss_energy[e], 0.0, True)
        if model == ModelsCsDiss.RS_THRESH:
            return self.integral_diss_RS(T, degree, interaction.collision_mass, interaction.collision_diameter, molecule.diss_energy[e], 0.0, False)
        if model == ModelsCsDiss.VSS_THRESH_CMASS_VIBR:
            if not interaction.vss_data:
                raise ModelParameterException("No VSS data for interaction")
            return self.integral_diss_VSS(T, degree, interaction.collision_mass, interaction.vss_c_cs, interaction.vss_omega, molecule.diss_energy[e], molecule.vibr_energy[e][i], True)
        if model == ModelsCsDiss.VSS_THRESH_VIBR:
            if not interaction.vss_data:
                raise ModelParameterException("No VSS data for interaction")
            return self.integral_diss_VSS(T, degree, interaction.collision_mass, interaction.vss_c_cs, interaction.vss_omega, molecule.diss_energy[e], molecule.vibr_energy[e][i], False)
        if model == ModelsCsDiss.VSS_THRESH_CMASS:
            if not interaction.vss_data:
                raise ModelParameterException("No VSS data for interaction")
            return self.integral_diss_VSS(T, degree, interaction.collision_mass, interaction.vss_c_cs, interaction.vss_omega, molecule.diss_energy[e], 0.0, True)
        if model == ModelsCsDiss.VSS_THRESH:
            if not interaction.vss_data:
                raise ModelParameterException("No VSS data for interaction")
            return self.integral_diss_VSS(T, degree, interaction.collision_mass, interaction.vss_c_cs, interaction.vss_omega, molecule.diss_energy[e], 0.0, False)
        if model == ModelsCsDiss.ILT:
            if e == 0:
                if molecule.name == "N2" and ("N" in (interaction.particle1_name, interaction.particle2_name)):
                    return self.integral_diss_ILT_N2N(T, degree, interaction.collision_mass, molecule.vibr_energy[e][i], self.convert_vibr_ladder_N2(molecule.vibr_energy[e][i]))
                if molecule.name == "O2" and ("O" in (interaction.particle1_name, interaction.particle2_name)):
                    return self.integral_diss_ILT_O2O(T, degree, interaction.collision_mass, molecule.vibr_energy[e][i], self.convert_vibr_ladder_O2(molecule.vibr_energy[e][i]))
            raise ModelParameterException("ILT dissociation available only for ground electronic state N2+N or O2+O")
        raise ModelParameterException(f"Unsupported dissociation integral model: {model}")

    def diss_rate_N2N_p4e(self, T: float, i: float) -> float:
        a = np.zeros(5)
        for j in range(5):
            a[j] = (
                self.p4e_n2n_diss_bjk[0][j]
                + self.p4e_n2n_diss_bjk[1][j] * i
                + self.p4e_n2n_diss_bjk[2][j] * (i * i)
                + self.p4e_n2n_diss_bjk[3][j] * (i * i * i)
            )
        return 1e-6 * math.exp(a[0] + a[1] / T + a[2] / (T * T) + a[3] / (T * T * T) + a[4] * math.log(T))

    def diss_rate_O2O_p4e(self, T: float, i: float) -> float:
        a = np.zeros(3)
        for j in range(3):
            a[j] = (
                self.p4e_o2o_diss_bjk[j][0]
                + self.p4e_o2o_diss_bjk[j][1] * i
                + self.p4e_o2o_diss_bjk[j][2] * (i * i)
                + self.p4e_o2o_diss_bjk[j][3] * (i * i * i)
                + self.p4e_o2o_diss_bjk[j][4] * (i**4)
                + self.p4e_o2o_diss_bjk[j][5] * (i**5)
                + self.p4e_o2o_diss_bjk[j][6] * (i**6)
                + self.p4e_o2o_diss_bjk[j][7] * (i**7)
            )
        return 1e-6 * math.exp(a[0] + a[1] / T + a[2] * math.log(T))

    def k_diss(self, T: float, molecule: Molecule, interaction: Interaction, i: int, e: int, model: ModelsKDiss, num_electron_levels: int = 1) -> float:
        if model == ModelsKDiss.RS_THRESH_CMASS_VIBR:
            return self.k_diss_RS(T, interaction.collision_mass, interaction.collision_diameter, molecule.diss_energy[e], molecule.vibr_energy[e][i], True)
        if model == ModelsKDiss.RS_THRESH_VIBR:
            return self.k_diss_RS(T, interaction.collision_mass, interaction.collision_diameter, molecule.diss_energy[e], molecule.vibr_energy[e][i], False)
        if model == ModelsKDiss.RS_THRESH_CMASS:
            return self.k_diss_RS(T, interaction.collision_mass, interaction.collision_diameter, molecule.diss_energy[e], 0.0, True)
        if model == ModelsKDiss.RS_THRESH:
            return self.k_diss_RS(T, interaction.collision_mass, interaction.collision_diameter, molecule.diss_energy[e], 0.0, False)
        if model == ModelsKDiss.VSS_THRESH_CMASS_VIBR:
            if not interaction.vss_data:
                raise DataNotFoundException(f"No VSS data found for {interaction.particle1_name} + {interaction.particle2_name} interaction")
            return self.k_diss_VSS(T, interaction.collision_mass, interaction.vss_c_cs, interaction.vss_omega, molecule.diss_energy[e], molecule.vibr_energy[e][i], True)
        if model == ModelsKDiss.VSS_THRESH_VIBR:
            if not interaction.vss_data:
                raise DataNotFoundException(f"No VSS data found for {interaction.particle1_name} + {interaction.particle2_name} interaction")
            return self.k_diss_VSS(T, interaction.collision_mass, interaction.vss_c_cs, interaction.vss_omega, molecule.diss_energy[e], molecule.vibr_energy[e][i], False)
        if model == ModelsKDiss.VSS_THRESH_CMASS:
            if not interaction.vss_data:
                raise DataNotFoundException(f"No VSS data found for {interaction.particle1_name} + {interaction.particle2_name} interaction")
            return self.k_diss_VSS(T, interaction.collision_mass, interaction.vss_c_cs, interaction.vss_omega, molecule.diss_energy[e], 0.0, True)
        if model == ModelsKDiss.VSS_THRESH:
            if not interaction.vss_data:
                raise DataNotFoundException(f"No VSS data found for {interaction.particle1_name} + {interaction.particle2_name} interaction")
            return self.k_diss_VSS(T, interaction.collision_mass, interaction.vss_c_cs, interaction.vss_omega, molecule.diss_energy[e], 0.0, False)
        if model == ModelsKDiss.ILT:
            if e != 0:
                raise ModelParameterException("No ILT model available for dissociation from excited electronic states")
            if ((interaction.particle1_name == "N2" and interaction.particle2_name == "N") or (interaction.particle1_name == "N" and interaction.particle2_name == "N2")):
                return self.k_diss_ILT_N2N(T, interaction.collision_mass, molecule.vibr_energy[e][i], self.convert_vibr_ladder_N2(molecule.vibr_energy[e][i]))
            if ((interaction.particle1_name == "O2" and interaction.particle2_name == "O") or (interaction.particle1_name == "O" and interaction.particle2_name == "O2")):
                return self.k_diss_ILT_O2O(T, interaction.collision_mass, molecule.vibr_energy[e][i], self.convert_vibr_ladder_O2(molecule.vibr_energy[e][i]))
            raise ModelParameterException(f"No ILT model available for dissociation for the interaction: {interaction.particle1_name}+{interaction.particle2_name}")
        if model == ModelsKDiss.ARRH_SCANLON:
            return self.k_Arrhenius(T, interaction[f"diss,{molecule.name},Arrh_A,Scanlon"], interaction[f"diss,{molecule.name},Arrh_n,Scanlon"], interaction[f"diss,{molecule.name},Ea,Scanlon"])
        if model == ModelsKDiss.ARRH_PARK:
            return self.k_Arrhenius(T, interaction[f"diss,{molecule.name},Arrh_A,Park"], interaction[f"diss,{molecule.name},Arrh_n,Park"], interaction[f"diss,{molecule.name},Ed,Park"])

        def _resolve_levels() -> int:
            if num_electron_levels == -1:
                return molecule.num_electron_levels
            return num_electron_levels

        def _arrhenius(prefix: str, key_energy: str) -> float:
            return self.k_Arrhenius(T, interaction[f"diss,{molecule.name},Arrh_A,{prefix}"], interaction[f"diss,{molecule.name},Arrh_n,{prefix}"], interaction[f"diss,{molecule.name},{key_energy},{prefix}"])

        is_scanlon = model in (ModelsKDiss.TM_D6K_ARRH_SCANLON, ModelsKDiss.TM_3T_ARRH_SCANLON, ModelsKDiss.TM_INF_ARRH_SCANLON, ModelsKDiss.TM_INFTY_ARRH_SCANLON)
        is_tm = model in (
            ModelsKDiss.TM_D6K_ARRH_SCANLON,
            ModelsKDiss.TM_3T_ARRH_SCANLON,
            ModelsKDiss.TM_INF_ARRH_SCANLON,
            ModelsKDiss.TM_INFTY_ARRH_SCANLON,
            ModelsKDiss.TM_D6K_ARRH_PARK,
            ModelsKDiss.TM_3T_ARRH_PARK,
            ModelsKDiss.TM_INF_ARRH_PARK,
            ModelsKDiss.TM_INFTY_ARRH_PARK,
        )
        if is_tm:
            prefix = "Scanlon" if is_scanlon else "Park"
            key_energy = "Ea" if is_scanlon else "Ed"
            k_arrh = _arrhenius(prefix, key_energy)
            levels = _resolve_levels()
            if model in (ModelsKDiss.TM_D6K_ARRH_SCANLON, ModelsKDiss.TM_D6K_ARRH_PARK):
                U = molecule.diss_energy[e] / (6 * constants.K_CONST_K)
                if levels == 1:
                    return k_arrh * self.Z_diss_U_vibr(T, U, molecule.vibr_energy[e], i)
                return k_arrh * self.Z_diss_U(T, U, molecule.electron_energy, molecule.statistical_weight, levels, molecule.vibr_energy, i, e)
            if model in (ModelsKDiss.TM_3T_ARRH_SCANLON, ModelsKDiss.TM_3T_ARRH_PARK):
                U = 3 * T
                if levels == 1:
                    return k_arrh * self.Z_diss_U_vibr(T, U, molecule.vibr_energy[e], i)
                return k_arrh * self.Z_diss_U(T, U, molecule.electron_energy, molecule.statistical_weight, levels, molecule.vibr_energy, i, e)
            if model in (ModelsKDiss.TM_INF_ARRH_SCANLON, ModelsKDiss.TM_INFTY_ARRH_SCANLON, ModelsKDiss.TM_INF_ARRH_PARK, ModelsKDiss.TM_INFTY_ARRH_PARK):
                if levels == 1:
                    return k_arrh * self.Z_diss_noU_vibr(T, molecule.vibr_energy[e], molecule.num_vibr_levels[e], i)
                return k_arrh * self.Z_diss_noU(T, molecule.electron_energy, molecule.statistical_weight, levels, molecule.vibr_energy, molecule.num_vibr_levels, i, e)

        if model == ModelsKDiss.PHYS4ENTRY:
            if e == 0:
                if molecule.name == "N2" and (interaction.particle1_name == "N" or interaction.particle2_name == "N"):
                    return self.diss_rate_N2N_p4e(T, self.convert_vibr_ladder_N2(molecule.vibr_energy[e][i]))
                if molecule.name == "O2" and (interaction.particle1_name == "O" or interaction.particle2_name == "O"):
                    return self.diss_rate_O2O_p4e(T, self.convert_vibr_ladder_O2(molecule.vibr_energy[e][i]))
                raise ModelParameterException(f"No Phys4Entry data for dissociation of {molecule.name} in reaction: {interaction.particle1_name}+{interaction.particle2_name}")
            raise ModelParameterException("Cannot compute dissociation rate using Phys4Entry data for excited electronic states")
        raise ModelParameterException(f"Unsupported dissociation rate model: {model}")

    def Z_coll(self, T: float, n: float, interaction: Interaction) -> float:
        return self.p_Z_coll(T, n, interaction.collision_mass, interaction.collision_diameter)

    def Boltzmann_distribution(self, T: float, n: float, molecule: Molecule, e: int) -> np.ndarray:
        if T <= 0:
            raise ModelParameterException("Temperature must be positive for Boltzmann distribution")
        if n <= 0:
            levels = molecule.num_vibr_levels[e] if molecule.num_vibr_levels else 1
            return np.zeros(levels)
        if e < 0 or e >= molecule.num_electron_levels:
            raise ModelParameterException("Invalid electronic level for Boltzmann distribution")
        return n * np.exp(-molecule.vibr_energy[e] / (constants.K_CONST_K * T)) / self.p_Z_vibr_eq(T, molecule.vibr_energy[e])

    @staticmethod
    def p_vibr_relaxation_time_MW(T: float, n: float, char_vibr_temp: float, coll_mass: float) -> float:
        md = coll_mass * constants.K_CONST_NA * 1000  # reduced molar mass, g/mol
        return pow(
            10.0,
            (5e-4 * pow(md, 0.5) * pow(char_vibr_temp, 4.0 / 3.0) * (pow(T, -1.0 / 3.0) - 0.015 * pow(md, 0.25)) - 8),
        ) / (n * constants.K_CONST_K * T * 9.8692326671601e-6)

    @staticmethod
    def p_vibr_relaxation_time_Park_corr(T: float, n: float, coll_mass: float, crosssection: float) -> float:
        return 1.0 / (math.sqrt(4 * constants.K_CONST_K * T / (constants.K_CONST_PI * coll_mass)) * n * crosssection)

    def vibr_relaxation_time_MW(self, T: float, n: float, molecule: Molecule, interaction: Interaction) -> float:
        return self.p_vibr_relaxation_time_MW(T, n, molecule.characteristic_vibr_temperatures[0], interaction.collision_mass)

    def vibr_relaxation_time_Park_corr(self, T: float, n: float, interaction: Interaction, crosssection: float) -> float:
        return self.p_vibr_relaxation_time_Park_corr(T, n, interaction.collision_mass, crosssection)

    def omega_integral(
        self,
        temperature: float,
        interaction: Interaction,
        l: int,
        r: int,
        model: ModelsOmega,
        dimensional: bool = True,
    ) -> float:
        if model == ModelsOmega.RS:
            val = self.omega_integral_RS(temperature, l, r, interaction.collision_diameter, interaction.collision_mass)
            return val if dimensional else 1.0
        if model == ModelsOmega.VSS:
            if not interaction.vss_data:
                raise ValueError(f"No VSS data found for {interaction.particle1_name}+{interaction.particle2_name}")
            val = self.omega_integral_VSS(temperature, l, r, interaction.collision_mass, interaction.vss_c_cs, interaction.vss_omega, interaction.vss_alpha)
            return val if dimensional else val / self.omega_integral_RS(temperature, l, r, interaction.collision_diameter, interaction.collision_mass)
        if model == ModelsOmega.LENNARD_JONES:
            base = lambda temp, l_val, r_val: self.omega_integral_LennardJones(temp, l_val, r_val, interaction.epsilon)
            return self._omega_integral_bracket(temperature, l, r, base, interaction, dimensional)
        if model == ModelsOmega.BORNMAYER:
            base = lambda temp, l_val, r_val: self.omega_integral_Born_Mayer(
                temp, l_val, r_val, interaction["beta Born-Mayer"], interaction["phi_zero Born-Mayer"], interaction.collision_diameter, interaction.epsilon
            )
            return self._omega_integral_bracket(temperature, l, r, base, interaction, dimensional)
        if model == ModelsOmega.ESA:
            def scale(val: float) -> float:
                return val * self.omega_integral_RS(temperature, l, r, interaction.collision_diameter, interaction.collision_mass) if dimensional else val

            # Hydrogen-specific fits
            if interaction.particle1_name in ("H2", "H") and interaction.particle2_name in ("H2", "H"):
                if interaction.particle1_name == "H2" and interaction.particle2_name == "H2":
                    return scale(self.omega_integral_ESA_H2H2(temperature, l, r, interaction.collision_diameter))
                if (interaction.particle1_name == "H2" and interaction.particle2_name == "H") or (interaction.particle1_name == "H" and interaction.particle2_name == "H2"):
                    return scale(self.omega_integral_ESA_H2H(temperature, l, r, interaction.collision_diameter))
                return scale(self.omega_integral_ESA_HH(temperature, l, r, interaction.collision_diameter))
            # Generic ESA using Bruno coefficients when available
            if interaction.charge1 == 0 and interaction.charge2 == 0:
                return scale(self.omega_integral_ESA_nn(temperature, l, r, interaction["beta Bruno"], interaction["epsilon0 Bruno"], interaction["re Bruno"]))
            if (interaction.charge1 == 0) ^ (interaction.charge2 == 0):
                base_val = self.omega_integral_ESA_cn(temperature, l, r, interaction["beta Bruno"], interaction["epsilon0 Bruno"], interaction["re Bruno"])
                corr_map = {
                    ("N", "N+"): {
                        (1, 1): (65.8423, -4.5492, 7.8608e-2),
                        (1, 2): (64.3259, -4.4969, 7.8619e-2),
                        (1, 3): (63.2015, -4.4575, 7.8613e-2),
                        (1, 4): (62.3098, -4.4260, 7.8606e-2),
                        (1, 5): (61.5722, -4.3997, 7.8601e-2),
                    },
                    ("Ar", "Ar+"): {
                        (1, 1): (68.6279, -4.3366, 6.8542e-2),
                        (1, 2): (67.1824, -4.2908, 6.8533e-2),
                        (1, 3): (66.1106, -4.2568, 6.8542e-2),
                        (1, 4): (65.2603, -4.2297, 6.8562e-2),
                        (1, 5): (64.5529, -4.2062, 6.8523e-2),
                    },
                    ("O", "O+"): {
                        (1, 1): (64.7044, -4.1936, 6.7974e-2),
                        (1, 2): (63.3075, -4.1485, 6.7991e-2),
                        (1, 3): (62.2707, -4.1146, 6.8000e-2),
                        (1, 4): (61.4470, -4.0872, 6.7986e-2),
                        (1, 5): (60.7663, -4.0646, 6.7987e-2),
                    },
                    ("C", "C+"): {
                        (1, 1): (65.8583, -4.8063, 8.7735e-2),
                        (1, 2): (64.2555, -4.7476, 8.7729e-2),
                        (1, 3): (63.0684, -4.7037, 8.7724e-2),
                        (1, 4): (62.1279, -4.6687, 8.7729e-2),
                        (1, 5): (61.3500, -4.6395, 8.7730e-2),
                    },
                    ("CO", "CO+"): {
                        (1, 1): (85.0889, -5.5980, 9.2122e-2),
                        (1, 2): (83.2212, -5.5362, 9.2097e-2),
                        (1, 3): (81.8376, -5.4902, 9.2105e-2),
                        (1, 4): (80.7381, -5.4530, 9.2078e-2),
                        (1, 5): (79.8298, -5.4225, 9.2091e-2),
                    },
                }
                pair = tuple(sorted((interaction.particle1_name, interaction.particle2_name)))
                correction = 0.0
                if pair in corr_map and (l, r) in corr_map[pair]:
                    d1, d2, d3 = corr_map[pair][(l, r)]
                    correction = self.omega_integral_ESA_cn_corr(temperature, d1, d2, d3, interaction["beta Bruno"], interaction["re Bruno"])
                if correction:
                    base_val = math.sqrt(base_val * base_val + correction * correction)
                return scale(base_val)
            # charged-charged (cc)
            return scale(self.omega_integral_ESA_cc(temperature, l, r, interaction.charge1, interaction.charge2, interaction.collision_diameter))
        raise ModelParameterException(f"omega_integral for model {model} is not implemented")

    def omega_integral_with_debye(
        self,
        temperature: float,
        interaction: Interaction,
        l: int,
        r: int,
        debye_length: float,
        model: ModelsOmega,
        dimensional: bool = True,
    ) -> float:
        if model == ModelsOmega.ESA and interaction.charge1 != 0 and interaction.charge2 != 0:
            val = self.omega_integral_ESA_cc(temperature, l, r, interaction.charge1, interaction.charge2, debye_length)
            if dimensional:
                val *= self.omega_integral_RS(temperature, l, r, interaction.collision_diameter, interaction.collision_mass)
            return val
        return self.omega_integral(temperature, interaction, l, r, model, dimensional)

    # ---------------- Rotational relaxation (Parker) ---------------- #
    @staticmethod
    def p_rot_collision_number_parker(T: float, xi_inf: float, epsilon: float) -> float:
        kTe = T * constants.K_CONST_K / epsilon
        return xi_inf / (1 + pow(constants.K_CONST_PI, 1.5) * pow(kTe, -0.5) / 2 + (constants.K_CONST_PI * constants.K_CONST_PI / 4 + 2) / kTe + pow(constants.K_CONST_PI, 1.5) * pow(kTe, -1.5))

    def rot_collision_number_parker(self, T: float, molecule: Molecule, interaction: Interaction) -> float:
        return self.p_rot_collision_number_parker(T, molecule.parker_const, interaction.epsilon)

    def rot_relaxation_time_parker(
        self,
        temperature: float,
        number_density: float,
        molecule: Molecule,
        interaction: Interaction,
        model: ModelsOmega,
    ) -> float:
        if temperature <= 0 or number_density <= 0:
            raise ValueError("Temperature and number density must be positive")
        if interaction.particle1_name != molecule.name and interaction.particle2_name != molecule.name:
            raise ValueError("Interaction does not involve the given molecule")
        xi = self.p_rot_collision_number_parker(temperature, molecule.parker_const, interaction.epsilon)
        return xi * constants.K_CONST_PI * 0.15625 / (number_density * self.omega_integral(temperature, interaction, 2, 2, model, True))
