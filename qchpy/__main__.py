import argparse
from pathlib import Path

from .basis_set import BASIS_SETS_FILENAMES
from .hf import prepare_rhf, rhf

#-------------INPUT PARAMETRES------------------------------------------------------------------------------------------
INPUT_FILE: str         = None             # Input geometry file in .xyz format
BASIS_TYPE: str         = "STO-2G"         # Basis set type used for computation

SHOW_BASIS_TYPES: bool  = False            # If set to True, the program returns available basis sets type without running the program

N_ELEC: int | None      = None             # The total number of the electrons, must be even number. If None is
                                           # specified then Z number of electron is chosen for every atom
E_TOL: float            = 1e-8             # Maximum energy error tolerance |dE| < E_TOL between iterations used in the SCF cycle
P_TOL: float            = 1e-6             # Maximum density error tolerance p_rms < P_TOL between iterations used in the SCF cycle
MAXIT: int              = 50               # Maximum number of iterations of the SCF cycle
DAMPING: float          = 0.1              # Value of damping used in the SCF cycle, new density is computed as: P = (1.0 - DAMPING)*P_new + DAMPING*P_old
OUTPUT_FILE: str | None = None             # Output file of the program. If None is specified, then case.out output file is created

RHF: bool              = True             # If set to True, the program runs restricted Hartree-Fock (RHF) calculations
#-----------------------------------------------------------------------------------------------------------------------

#------------AVAILABLE BASIS SETS---------------------------------------------------------------------------------------
# STO-2G   STO-3G   STO-4G   STO-5G   STO-6G
# 3-21G    6-21G    6-31G    6-311G
# 6-31++G  6-31++G**         6-311+G
#-----------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--input", default = INPUT_FILE, type = str, help = "Input geometry file in .xyz format")
parser.add_argument("--basis", default = BASIS_TYPE, type = str, help = "Basis set type used for computation")
parser.add_argument("--showb", default = SHOW_BASIS_TYPES, action = "store_true", help = "If set to true, the program returns available basis sets type without running the computations")
parser.add_argument("--nelec", default = N_ELEC, type = int, help = "The total number of the electrons, must be even number. If None is specified then Z number of electron is chosen for every atom")
parser.add_argument("--etol", default = E_TOL, type = float, help = "Maximum energy error tolerance |dE| < etol between iterations used in the SCF cycle")
parser.add_argument("--ptol", default = P_TOL, type = float, help = "Maximum density error tolerance p_rms < ptol between iterations used in the SCF cycle")
parser.add_argument("--maxit", default = MAXIT, type = int, help = "Maximum number of iterations of the SCF cycle")
parser.add_argument("--damping", default = DAMPING, type = float, help = "Value of damping used in the SCF cycle, new density is then computed as: P = (1.0 - damping)*P_new + damping*P_old")
parser.add_argument("--output", default = OUTPUT_FILE, type = str, help = "Output file of the program. If None is specified, then case.out output file is created")
parser.add_argument("--rhf", default = RHF, action = "store_true", help = "If set to true, the program runs restricted Hartree-Fock (RHF) calculations")

if __name__ == "__main__":

    args = parser.parse_args([] if "__file__" not in globals() else None)

    if args.showb:
        print("AVAILABLE BASIS SET TYPES:")
        for basis, _ in BASIS_SETS_FILENAMES.items():
            print(basis)
        exit()

    if not args.input:
        raise RuntimeError("You need to specify input .xyz file, either in the file itself or through the argument parser!")

    if args.basis.upper() not in BASIS_SETS_FILENAMES:
        raise RuntimeError(f"I do not know this basis set type yet: {args.basis}. Try running the program with --showb=True to see the available basis set types")

    if args.rhf:
        basis, atoms, n_elec = prepare_rhf(args.input, args.basis, args.nelec)
        rhf(basis.cgtos, atoms, n_elec, args.maxit, args.damping, args.etol, args.ptol)