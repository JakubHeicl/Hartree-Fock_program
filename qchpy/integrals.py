import numpy as np
from tqdm import tqdm

from .gaussian_calc import ContractedGaussian, twoel_cgto, kinetic_cgto, nucatr_cgto, overlap_cgto

def build_eri(cgtos: list[ContractedGaussian], eri_thresh: float = 1e-10) -> np.ndarray:
    n = len(cgtos)
    eri = np.zeros((n, n, n, n), dtype=np.float64)

    def pair_index(a, b):
        return a*(a+1)//2 + b
    
    npair = n * (n + 1) // 2
    bounds = np.zeros(npair)

    for i in range(n):
        for j in range(i + 1):
            pij = pair_index(i, j)
            bounds[pij] = np.sqrt(abs(twoel_cgto(cgtos[i], cgtos[j], cgtos[i], cgtos[j])))

    for i in range(n):
        for j in range(i+1):
            pij = pair_index(i, j)
            for k in range(n):
                for l in range(k+1):
                    pkl = pair_index(k, l)
                    if pij < pkl:
                        continue
                    if bounds[pij] * bounds[pkl] < eri_thresh:
                        continue

                    v = twoel_cgto(cgtos[i], cgtos[j], cgtos[k], cgtos[l])

                    eri[i,j,k,l] = v
                    eri[j,i,k,l] = v
                    eri[i,j,l,k] = v
                    eri[j,i,l,k] = v
                    eri[k,l,i,j] = v
                    eri[l,k,i,j] = v
                    eri[k,l,j,i] = v
                    eri[l,k,j,i] = v
    return eri

def build_h(cgtos: list[ContractedGaussian], atoms: list[tuple[int, np.ndarray]]) -> np.ndarray:

    H = np.zeros((len(cgtos), len(cgtos)))

    for i, A in enumerate(cgtos):
        for j, B in enumerate(cgtos):

            H[i][j] = kinetic_cgto(A, B)

            for atom in atoms:
                Z, coor = atom

                H[i][j] += -Z*nucatr_cgto(A, B, coor)
    
    return H

def build_S(cgtos: list[ContractedGaussian]):

    S = np.zeros((len(cgtos), len(cgtos)))

    for i, A in enumerate(cgtos):
        for j, B in enumerate(cgtos):
            S[i][j] = overlap_cgto(A, B)

    return S

def build_Enuc(atoms: list[tuple[int, np.ndarray]]) -> float:
    Enuc = 0.0
    for a in range(len(atoms)):
        Za, Ra = atoms[a]
        for b in range(a + 1, len(atoms)):
            Zb, Rb = atoms[b]
            Rab = np.linalg.norm(Ra - Rb)
            if Rab < 1e-12:
                raise ValueError("Two nuclei are at the same position (R_AB ~ 0).")
            Enuc += Za * Zb / Rab

    return Enuc