#!/bin/bash

TOOL_NAME=alpha-beta-CROWN
VERSION_STRING=v1
if [[ -z "${VNNCOMP_PYTHON_PATH}" ]]; then
	VNNCOMP_PYTHON_PATH=/home/ubuntu/anaconda3/envs/pytorch_latest_p37/bin
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

# Fixup permission problems, remove residual files.
echo "Fixing up permissions..."
sudo rm -rf ${VNNCOMP_PYTHON_PATH}/../lib/python3.7/site-packages/\~nnx2pytorch*  # This folder wasn't delete successfully due to permission issues.
sudo chown -R ubuntu:ubuntu ${VNNCOMP_PYTHON_PATH}/../../  # Some packages were mistakenly installed by sudo. Change their permission back.
sudo rm -rf ${HOME}/.local/share/auto_LiRPA/
sudo chown -R ubuntu:ubuntu ${HOME}

# Force uninstall the onnx2pytorch library if it is already installed.
${VNNCOMP_PYTHON_PATH}/python -m pip uninstall -y onnx2pytorch

# Install requirements.
sudo rm ${HOME}/vnncomp_requirements.txt
cat << 'EOF' > ${HOME}/vnncomp_requirements.txt
numpy>=1.16,<1.20
packaging>=20.0
pytest>=5.0
appdirs>=1.4
oslo.concurrency>=4.2
tqdm>=4.6
sortedcontainers>=2.4
onnx==1.9.0
onnxruntime==1.8.0
git+git://github.com/Sarimuko/onnx2pytorch@8879f72c41ba960ff6495ae754d885eac2ebf656#egg=onnx2pytorch
EOF
${VNNCOMP_PYTHON_PATH}/python -m pip install -r ${HOME}/vnncomp_requirements.txt

# Install our auto_LiRPA library.
cd ${TOOL_DIR}
${VNNCOMP_PYTHON_PATH}/python setup.py develop

echo "Checking if installation works by runnning a tiny network..."
temp_file=$(mktemp)
${VNNCOMP_PYTHON_PATH}/python3 ${TOOL_DIR}/src/bab_verification_general.py --data TEST --onnx_path ${TOOL_DIR}/src/tests/test_tiny.onnx --vnnlib_path ${TOOL_DIR}/src/tests/test_tiny.vnnlib --results_file ${temp_file} --timeout 300.0 --pgd_order skip

# Make sure we installed the right onnx2pytorch version.
echo "Checking if we have a working onnx2pytorch package..."
wget -q -O /tmp/sigmoid.onnx https://github.com/stanleybak/vnncomp2021/raw/main/benchmarks/eran/nets/ffnnSIGMOID__Point_6x200.onnx
wget -q -O /tmp/sigmoid.vnnlib https://raw.githubusercontent.com/stanleybak/vnncomp2021/main/benchmarks/eran/specs/mnist/mnist_spec_idx_7167_eps_0.01200.vnnlib
# Check the model prediction to see if it matches our expectation. If not, the onnx2pytorch package is buggy.
prediction=$(${VNNCOMP_PYTHON_PATH}/python3 ${TOOL_DIR}/src/bab_verification_general.py --data MNIST --onnx_path /tmp/sigmoid.onnx --vnnlib_path /tmp/sigmoid.vnnlib --results_file ${temp_file} | grep "Model prediction is" | grep -Eo '[+-]?[0-9]+([.][0-9]+)?' | head -n 1)
if [ "$prediction" != "-21.6972" ]; then
    echo "onnx2pytorch is buggy. Please remove any existing onnx2pytorch package or run in a freshly created AWS instance."
    exit 1
else
    echo "onnx2pytorch checking PASSED!"
fi
rm $temp_file


# Install Gurobi.
echo "Installing Gurobi. It might take a while..."
conda install -c gurobi -n pytorch_latest_p37 -y gurobi
# Make sure basic pytorch runs.
echo "Checking if pytorch works... (returns 1.0 == works, it might take a while...)"
${VNNCOMP_PYTHON_PATH}/python -c 'import torch; a=torch.ones(1, device="cuda"); print(a.item())'

# Activate Gurobi with academic license.
echo "Please enter Gurobi licence (press ctrl+C to skip if gurobi license is already installed):"
read gurobi_key
${VNNCOMP_PYTHON_PATH}/grbgetkey --path ${HOME} -q ${gurobi_key}

