from pathlib import Path
import numpy as np

from .scf import scf_rhf
from .utils import read_xyz
from .basis_set import build_basis_set
from .gaussian_calc import ContractedGaussian
from .integrals import build_eri, build_h, build_S, build_Enuc

def rhf(cgtos: list[ContractedGaussian], atoms: list[tuple[int, np.ndarray]], n_elec: int, 
        maxiter: int = 50, 
        damping: float = 0.1, 
        e_tol: float = 1e-8, 
        p_rms_tol: float = 1e-6) -> None:

    eri = build_eri(cgtos)
    S = build_S(cgtos)
    H = build_h(cgtos, atoms)
    enuc = build_Enuc(atoms)

    converged, iteration, E_tot, eps, C = scf_rhf(H, S, eri, enuc, n_elec, maxiter, damping, e_tol, p_rms_tol)

def prepare_rhf(xyz_filename: Path, basis_type: str, n_elec: int = None):

    _, Zs, coors = read_xyz(xyz_filename)

    atoms = []
    for Z, coor in zip(Zs, coors):
        atoms.append((Z, coor))

    if n_elec is None:
        n_elec = sum(Zs)

    if n_elec % 2 != 0:
        raise ValueError("Number of electrons must be even for RHF")

    if out_filename is None:
         stem = Path(xyz_filename).stem
         out_filename = f"{stem}.out"

    basis = build_basis_set(atoms, basis_type)

    return basis, atoms, n_elec