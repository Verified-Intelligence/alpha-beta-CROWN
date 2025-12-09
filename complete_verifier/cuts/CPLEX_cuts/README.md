CPLEX Cuts
===================

A C++ example program to dump cuts from CPLEX solver. There is no existing python interface for this functionality.

In this example I have change CPLEX parameters to tell it to focus on cuts without branching, and without looking for primal solutions. The cut strength is similar to Gurobi and much stronger than scip.

Install CPLEX
-------

It's free for academic use, but the download procedure is complicated. So I uploaded the latest version here:

```
wget http://d.huan-zhang.com/storage/programs/cplex_studio2211.linux_x86_64.bin
chmod +x cplex_studio2211.linux_x86_64.bin
./cplex_studio2211.linux_x86_64.bin
```

Usage
-------

First change the cplex installation path in `Makefile`. Then run `make` to compile.

In the GCP-CROWN pipeline `./get_cuts` is run automatically by the verifier.

Example usage: `./get_cuts input_filename.mps output_filename`

The input .mps file is a standard format of saving a LP/MIP problem in human-readable form and our current Gurobi building code can generate it (see `_build_the_model_mip_mps_save()`). Example: `ALPHA_BETA_CROWN_MIP_DEBUG=1 python abcrown.py --config exp_configs/vnncomp21/oval21.yaml --select_instance 5`

Two output files will be saved: `output_filename.indx` and `output_filename.cuts`. `output_filename.indx` is only updated once, which contains the variable ID to variable name mapping. `output_filename.cuts` contains coefficients for all cuts.

In the GCP-CROWN pipeline, all the `.mps` files and their corresponding output files are stored in a temporary directory. 

You can also generate the corresponding `.mps` file with all cuts added, by uncommenting [lines 258-273 of `get_cuts.cpp`](get_cuts.cpp).
It is used for checking the correctness of cuts because it is human-readable, but the generated file is very large and cannot be efficiently parsed in python.


How to interpret results?
-------

Look at the generated `output_filename.mps`. All the constraints with letter `r`, `i`, `m`, `q`, `L` etc are cuts.
Coefficients are stored in sparse format. For example, you can search for `m1038` for all the coefficients of the 1038-th cut.

The `output_filename.indx` and `output_filename.cuts` files contain the same information as the `output_filename.mps` (~20MB, 300K lines) but they are much smaller and efficient to handle. See source code for definition of these binary files.

TODOs
-------

The easiest way is to generate the .mps file using our existing gurobi code and
find cuts using this program, see commit 60be115ed3232250fdb95102a882b9feef5efb6e.
This program should periodically output cuts
which are monitored by our verifier in a separated thread. Then, these cuts are
processed and added during bab.

The mps files may have numerical issues and are very large to read and process.
The next step is to generate the cuts directly in python by adding new python
bindings.

