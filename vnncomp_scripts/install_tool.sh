#!/bin/bash

# Installation script used for VNN-COMP. The tool is only compatible with the AWS Deep Learning AMI (Ubuntu 18.04) Version 46.0.
# For installation on a generic system, please use "install_tool_generic.sh" instead.

TOOL_NAME=alpha-beta-CROWN
VERSION_STRING=v1
if [[ -z "${VNNCOMP_PYTHON_PATH}" ]]; then
	VNNCOMP_PYTHON_PATH=/home/ubuntu/anaconda3/envs/alpha-beta-crown/bin
fi

# check arguments
if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi

echo "Installing $TOOL_NAME"
TOOL_DIR=$(dirname $(dirname $(realpath $0)))

echo "Checking system requirements..."
# Check System version.
if [ -f '/etc/update-motd.d/00-header' ]; then
    if ! grep 'Deep Learning AMI (Ubuntu 18.04) Version 46.0' /etc/update-motd.d/00-header >/dev/null; then
        echo "Unsupported OS! Deep Learning AMI (Ubuntu 18.04) Version 46.0 required."
        exit 1
    fi
else
    echo "Unsupported OS! We require Deep Learning AMI (Ubuntu 18.04) Version 46.0"
    echo "Please make sure you choose the right image and version when creating the instance."
    exit 1
fi
# Check PyTorch version.
echo "Checking python requirements (it might take a while...)"
if [ "$(${VNNCOMP_PYTHON_PATH}/python -c 'import torch; print(torch.__version__)')" != '1.8.1+cu111' ]; then
    echo "Unsupported PyTorch version - we expect to run on Amazon Deep Learning AMI 46.0"
    echo "Installation Failure!"
    exit 1
fi

# Turnoff useless programs.
sudo snap remove amazon-ssm-agent
sudo systemctl stop unattended-upgrades.service docker.service containerd.service snapd.service
sudo systemctl disable unattended-upgrades.service docker.service containerd.service docker.socket snapd.service snapd.socket

# Install requirements.
cat << 'EOF' > ${HOME}/vnncomp_requirements.txt
numpy>=1.16
packaging>=20.0
pytest>=5.0
appdirs>=1.4
oslo.concurrency>=4.2
tqdm>=4.6
sortedcontainers>=2.4
onnx==1.9.0
onnxruntime==1.8.0
git+git://github.com/Sarimuko/onnx2pytorch@master#egg=onnx2pytorch
EOF
${VNNCOMP_PYTHON_PATH}/python -m pip install -r ${HOME}/vnncomp_requirements.txt
# Install our auto_LiRPA library.
cd ${TOOL_DIR}
${VNNCOMP_PYTHON_PATH}/python setup.py develop

echo "Checking if installation works by runnning a tiny network..."
temp_file=$(mktemp)
${VNNCOMP_PYTHON_PATH}/python3 ${TOOL_DIR}/src/bab_verification_general.py --data TEST --onnx_path ${TOOL_DIR}/src/tests/test_tiny.onnx --vnnlib_path ${TOOL_DIR}/src/tests/test_tiny.vnnlib --results_file ${temp_file} --timeout 300.0 --pgd_order skip
rm $temp_file


# Install Gurobi.
echo "Installing Gurobi. It might take a while..."
conda install -c gurobi -n pytorch_latest_p37 -y gurobi
# Make sure basic pytorch runs.
echo "Checking if pytorch works... (returns 1.0 == works, it might take a while...)"
${VNNCOMP_PYTHON_PATH}/python -c 'import torch; a=torch.ones(1, device="cuda"); print(a.item())'

# Activate Gurobi with academic license.
echo "Please enter Gurobi licence:"
read gurobi_key
${VNNCOMP_PYTHON_PATH}/grbgetkey --path ${HOME} -q ${gurobi_key}

