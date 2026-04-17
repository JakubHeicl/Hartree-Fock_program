import numpy as np
from tqdm import tqdm
from time import perf_counter

from .gaussian_calc import *
from .basis_set import *
from .integrals import build_eri, build_h, build_S

Array = np.ndarray

def solve_roothaan(F: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Fp = X.T @ F @ X
    eps, Cp = np.linalg.eigh(Fp)
    C = X @ Cp
    return eps, C

def build_X(S: np.ndarray, thresh: float = 1e-12) -> np.ndarray:
    s, U = np.linalg.eigh(S)
    if np.any(s < -1e-10):
        raise ValueError("Overlap matrix S has significantly negative eigenvalues (basis issue).")
    s_clipped = np.clip(s, thresh, None)
    X = (U * (s_clipped ** -0.5)) @ U.T

    return X

def build_P(C: np.ndarray, n_elec: int) -> np.ndarray:
    nocc = n_elec // 2
    Cocc = C[:, :nocc]
    P = 2.0 * (Cocc @ Cocc.T)
    return P

def build_F(P: np.ndarray, eri: np.ndarray, H: np.ndarray) -> np.ndarray:
    J = np.einsum("ls,mnls->mn", P, eri, optimize=True)

    K = np.einsum("ls,mlns->mn", P, eri, optimize=True)

    F = H + J - 0.5 * K
    F = 0.5 * (F + F.T)
    return F

def energy(H: np.ndarray, F: np.ndarray, P: np.ndarray) -> float:
    return 0.5 * np.einsum("mn,mn->", P, (H + F), optimize=True)

def scf_rhf(eri: np.ndarray, 
            H: np.ndarray, 
            S: np.ndarray, 
            enuc: float, 
            n_elec: int, 
            maxiter: int = 50, 
            damping: float = 0.1, 
            e_tol: float = 1e-8, 
            p_rms_tol: float = 1e-6) -> tuple[bool, int, float, np.ndarray, np.ndarray]:
    
    """
    Performs the SCF procedure for a closed-shell system.
    Args:
        eri: Electron repulsion integrals in the basis of contracted Gaussian functions (n_functions, n_functions, n_functions, n_functions)
        H: One-electron Hamiltonian matrix in the basis of contracted Gaussian functions (n_functions, n_functions)
        S: Overlap matrix in the basis of contracted Gaussian functions (n_functions, n_functions)
        enuc: Nuclear repulsion energy
        n_elec: Total number of electrons (must be even)
        maxiter: Maximum number of SCF iterations
        damping: Damping factor for density matrix mixing (0.0 means no damping)
        e_tol: Energy convergence tolerance (SCF converged if |dE| < e_tol)
        p_rms_tol: Density matrix convergence tolerance (SCF converged if p_rms < p_rms_tol)
    Returns:
        converged: True if SCF converged, False otherwise
        n_iterations: Number of iterations performed
        E_tot: Final total energy (electronic + nuclear)
        eps: Molecular orbital energies
        C: Molecular orbital coefficients
    """

    X = build_X(S)
    eps, C = solve_roothaan(H, X)
    P = build_P(C, n_elec)
    E_elec = energy(H, F, P)
    E_tot = E_elec + enuc

    converged = False

    for iteration in range(maxiter):

        F = build_F(P, eri, H)

        eps, C = solve_roothaan(F, X)

        P_new = build_P(C, n_elec)

        if damping > 0.0:
            P_new = (1.0 - damping) * P_new + damping * P

        E_elec_new = energy(H, F, P_new)
        E_tot_new = E_elec_new + enuc

        dP = P_new - P
        p_rms = np.sqrt(np.mean(dP * dP))

        dE = E_tot_new - E_tot

        P = P_new
        E_tot = E_tot_new

        if abs(dE) < e_tol and p_rms < p_rms_tol:
            converged = True
            break

    F_final = build_F(P, eri, H)
    eps, C = solve_roothaan(F_final, X)
    E_elec_final = energy(H, F_final, P)
    E_tot_final = E_elec_final + enuc

    return converged, iteration + 1, E_tot_final, eps, C