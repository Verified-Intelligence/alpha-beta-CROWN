import os
import sys
from setuptools import setup, find_packages
from pathlib import Path


pytorch_version_l = '2.0.0'
pytorch_version_u = '2.9.0'     # excluded
torchvision_version_l = '0.12.0'
torchvision_version_u = '0.24.0'  # excluded

msg_install_pytorch = (
    f'It is recommended to manually install PyTorch '
    f'(>={pytorch_version_l},<{pytorch_version_u}) suitable '
    f'for your system ahead of time: https://pytorch.org/get-started.\n'
)

try:
    import torch
    if torch.__version__ < pytorch_version_l:
        print(f'PyTorch version {torch.__version__} is too low. '
              + msg_install_pytorch)
    if torch.__version__ >= pytorch_version_u:
        print(f'PyTorch version {torch.__version__} is too high. '
              + msg_install_pytorch)
except ModuleNotFoundError:
    print(f'PyTorch is not installed. {msg_install_pytorch}')


version = None
init_file = Path("complete_verifier/__init__.py")
for line in init_file.read_text().splitlines():
    if line.strip().startswith("__version__"):
        version = eval(line.split("=")[-1].strip())
        break

if version is None:
    raise RuntimeError("Cannot find __version__ in complete_verifier/__init__.py")


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

cplex_package_files = ["get_cuts"]
get_cuts_path = os.path.join("complete_verifier", "cuts", "CPLEX_cuts", "get_cuts")
if not os.path.exists(get_cuts_path):
    print(f"\n{'!'*60}")
    print(f"WARNING: The binary 'get_cuts' was NOT found at:")
    print(f"         {get_cuts_path}")
    print(f"{'!'*60}\n")

    # Check if running interactively (User typing in terminal)
    if sys.stdin and sys.stdin.isatty():
        response = input("Do you want to continue installation anyway? (y/n): ")
        
        if response.lower() == 'y':
            print("User opted to continue without 'get_cuts'.")
            cplex_package_files = []
        else:
            print("Installation aborted by user.")
            sys.exit(1)     
    else:
        # Non-Interactive (pip install / CI script)
        # We default to proceeding with a warning so automated builds don't hang
        print("Non-interactive mode: Proceeding without 'get_cuts'.")
        cplex_package_files = []

print(f"Installing alpha-beta-CROWN {version}")

setup(
    name="abcrown",
    version=version,
    description=(
        "alpha-beta-CROWN: An Efficient, Scalable and GPU Accelerated "
        "Neural Network Verifier (winner of VNN-COMP 2021, 2022, 2023, 2024, 2025)"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Verified-Intelligence/alpha-beta-CROWN",
    author="α,β-CROWN Team",
    author_email="huan@huan-zhang.com, xiangru4@illinois.edu",
    packages=find_packages(),
    package_data={
        "abcrown.cuts.CPLEX_cuts": cplex_package_files,
    },
    include_package_data=True,
    install_requires=[
        f"torch>={pytorch_version_l},<{pytorch_version_u}",
        f"torchvision>={torchvision_version_l},<{torchvision_version_u}",
        "numpy>=1.20",
        "packaging>=20.0",
        "ninja>=1.10",
        "tqdm>=4.64",
        "graphviz>=0.20.3",
        "appdirs>=1.4",
        "pytest==8.1.1",
        "pytest-order>=1.0.0",
        "pytest-mock>=3.14",
        "pylint>=2.15",
        "onnxruntime>=1.15",
        "onnxsim>=0.4.31",
        "skl2onnx>=1.14",
        "termcolor>=2.3.0",
        "onnxoptimizer>=0.3",
        "gurobipy>=10",
        "psutil>=5.9.5",
        "pyyaml>=6.0",
        "sortedcontainers>=2.4",
        "pandas>=2.0",
        "onnx2pytorch @ git+https://github.com/Verified-Intelligence/onnx2pytorch.git"
    ],
    platforms=["any"],
    license="BSD",
)
