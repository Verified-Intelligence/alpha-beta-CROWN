#!/bin/bash

# Script for installing environments for a generic Conda environment.

conda create --yes --name alpha-beta-crown python=3.7
source activate alpha-beta-crown
conda install --yes pytorch torchvision torchaudio cudatoolkit=11.1 mkl=2020.0 gurobi pyyaml pytest packaging pandas tqdm appdirs protobuf sortedcontainers -c pytorch-lts -c nvidia -c gurobi
pip install --no-input --no-cache-dir onnx onnxruntime git+https://github.com/KaidiXu/onnx2pytorch.git git+https://github.com/dlshriver/DNNV.git@develop

# Activate Gurobi with academic license.
echo "Please enter Gurobi licence (optional):"
read gurobi_key
if [ -z "$gurobi_key" ]; then
    echo "Skipped."
else
    ${VNNCOMP_PYTHON_PATH}/grbgetkey --path ${HOME} -q ${gurobi_key}
fi

