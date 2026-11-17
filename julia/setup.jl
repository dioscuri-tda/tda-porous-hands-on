# Installs the Julia dependencies of the Euler characteristic profile (ECP) step:
#   julia --project=julia julia/setup.jl
# Ecp is not in the General registry; it is installed from GitHub at the commit
# used in the papers.
import Pkg
Pkg.add(url="https://github.com/Malinon/ECP_toolkit", rev="d110d1ebcb7078736468fdf9eee05328b6bc4691")
Pkg.add(["NPZ", "Glob"])
Pkg.instantiate()
