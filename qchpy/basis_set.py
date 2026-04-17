import numpy as np
from dataclasses import dataclass, field
import json
from pathlib import Path

from .utils import cartesian_tuples
from .gaussian_calc import *

BASIS_SETS_DIR = Path(__file__).parent / "basis_sets"

BASIS_SETS_FILENAMES = {"STO-2G":       Path(BASIS_SETS_DIR, "sto-2g.txt"),
                        "STO-3G":       Path(BASIS_SETS_DIR, "sto-3g.txt"),
                        "STO-4G":       Path(BASIS_SETS_DIR, "sto-4g.txt"),
                        "STO-5G":       Path(BASIS_SETS_DIR, "sto-5g.txt"),
                        "STO-6G":       Path(BASIS_SETS_DIR, "sto-6g.txt"),
                        "3-21G":        Path(BASIS_SETS_DIR, "3-21g.txt"),
                        "6-21G":        Path(BASIS_SETS_DIR, "6-21g.txt"),
                        "6-31G":        Path(BASIS_SETS_DIR, "6-31g.txt"),
                        "6-311G":       Path(BASIS_SETS_DIR, "6-311g.txt"),
                        "6-31++G":      Path(BASIS_SETS_DIR, "6-31++g.txt"),
                        "6-31++G**":    Path(BASIS_SETS_DIR, "6-31++gss.txt"),
                        "6-311+G":      Path(BASIS_SETS_DIR, "6-311+g.txt")}

@dataclass
class Basis:
    cgtos: list[ContractedGaussian] = field(default_factory=list)

    def add_cgto(self, cgto: ContractedGaussian) -> None:
        self.cgtos.append(cgto)

def build_basis_set(atoms: list[tuple[int, np.ndarray]], basis_type: str) -> Basis:
    
    if basis_type not in BASIS_SETS_FILENAMES:
        raise RuntimeError(f"I do not know this basis set type: {basis_type}")
    
    with open(BASIS_SETS_FILENAMES[basis_type], "r", encoding="utf-8") as f:
        basis_set_file = json.load(f)

    elements_file = basis_set_file.get("elements", {})

    bs = Basis()

    for atom in atoms:

        Z, atom_coor = atom

        edata = elements_file.get(str(Z))
        if edata is None:
            raise ValueError(f"Basis JSON does not contain element Z={Z}.")
        
        for shell in edata.get("electron_shells", []):
            gaussian_type = shell.get("function_type", "gto")

            if gaussian_type.lower() not in ["gto_cartesian", "gto"]:
                raise RuntimeError(f"I do not know this type of gaussian: {gaussian_type}")
            
            ls = [int(l) for l in shell["angular_momentum"]]    #list[int]
            exps = [float(exp) for exp in shell["exponents"]]   #list[float]
            coeffs_all = shell["coefficients"]                  #list[list[str]]

            for l, coeffs in zip(ls, coeffs_all):
                for cartesian_tuple in cartesian_tuples(l):

                    cgto = ContractedGaussian(atom_coor, cartesian_tuple)

                    for alpha, coeff in zip(exps, coeffs):

                        cgto.add_primitive(alpha, float(coeff))

                    bs.add_cgto(cgto)
    
    return bs