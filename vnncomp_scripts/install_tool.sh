#!/bin/bash

# Installation script used for VNN-COMP. The tool is only compatible with Ubuntu 24.04.

TOOL_NAME=alpha-beta-CROWN
VERSION_STRING=v1
if [[ -z "${VNNCOMP_PYTHON_PATH}" ]]; then
	VNNCOMP_PYTHON_PATH=/home/ubuntu/miniconda/envs/alpha-beta-crown/bin
fi

# check arguments
if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi

echo "Installing $TOOL_NAME"
TOOL_DIR=$(dirname $(dirname $(realpath $0)))

export DEBIAN_FRONTEND=noninteractive
sudo -E DEBIAN_FRONTEND=noninteractive apt purge -y snapd unattended-upgrades modemmanager
sudo killall -9 unattended-upgrade-shutdown
sudo -E DEBIAN_FRONTEND=noninteractive apt update
sudo -E DEBIAN_FRONTEND=noninteractive apt upgrade -y
sudo -E DEBIAN_FRONTEND=noninteractive apt install -y sudo vim-gtk3 curl wget git cmake tmux aria2 build-essential netcat-openbsd expect dkms aria2

sudo systemctl stop cron.service chrony.service multipathd.service multipathd.socket udisks2.service packagekit.service polkit.service networkd-dispatcher.service
sudo systemctl disable cron.service chrony.service multipathd.service multipathd.socket udisks2.service packagekit.service polkit.service networkd-dispatcher.service
sudo systemctl mask cron.service chrony.service multipathd.service multipathd.socket udisks2.service packagekit.service polkit.service networkd-dispatcher.service

grep AMD /proc/cpuinfo > /dev/null && echo "export MKL_DEBUG_CPU_TYPE=5" >> ${HOME}/.profile
echo "export OMP_NUM_THREADS=1" >> ${HOME}/.profile

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
sh miniconda.sh -b -p ${HOME}/miniconda
echo 'export PATH=${PATH}:'${HOME}'/miniconda/bin' >> ~/.profile
echo "alias py37=\"source activate alpha-beta-crown\"" >> ${HOME}/.profile
export PATH=${PATH}:$HOME/miniconda/bin

# Install NVIDIA driver
DRIVER_VERSION=550.78
aria2c -x 10 -s 10 -k 1M https://us.download.nvidia.com/XFree86/Linux-x86_64/$DRIVER_VERSION/NVIDIA-Linux-x86_64-$DRIVER_VERSION.run
sudo nvidia-smi -pm 0
chmod +x ./NVIDIA-Linux-x86_64-$DRIVER_VERSION.run
sudo ./NVIDIA-Linux-x86_64-$DRIVER_VERSION.run --silent --dkms
# Remove old driver (if already installed) and reload the new one.
sudo rmmod nvidia_uvm; sudo rmmod nvidia_drm; sudo rmmod nvidia_modeset; sudo rmmod nvidia
sudo modprobe nvidia; sudo nvidia-smi -e 0; sudo nvidia-smi -r -i 0
sudo nvidia-smi -pm 1
# Make sure GPU shows up.
nvidia-smi

# Install conda environment
${HOME}/miniconda/bin/conda env create --name alpha-beta-crown -f ${TOOL_DIR}/complete_verifier/environment_pyt231.yaml
${HOME}/miniconda/bin/conda env create --name alpha-beta-crown-2022 -f ${TOOL_DIR}/complete_verifier/environment_2022.yaml

# Install CPLEX
aria2c -x 10 -s 10 -k 1M "http://d.huan-zhang.com/storage/programs/cplex_studio2211.linux_x86_64.bin"
chmod +x cplex_studio2211.linux_x86_64.bin
cat > response.txt <<EOF
INSTALLER_UI=silent
LICENSE_ACCEPTED=true
EOF
sudo ./cplex_studio2211.linux_x86_64.bin -f response.txt

# Build CPLEX interface
make -C ${TOOL_DIR}/complete_verifier/cuts/CPLEX_cuts/

echo "Checking python requirements (it might take a while...)"
if [ "$(${VNNCOMP_PYTHON_PATH}/python -c 'import torch; print(torch.__version__)')" != '2.3.1' ]; then
    echo "Unsupported PyTorch version"
    echo "Installation Failure!"
    exit 1
fi


# Setup Gurobi
grbprobe_output=$(${VNNCOMP_PYTHON_PATH}/grbprobe)
echo $grbprobe_output

HOSTNAME=$(echo $grbprobe_output | grep -Po "(?<=HOSTNAME=)(.*?)(?= )")
HOSTID=$(echo $grbprobe_output | grep -Po "(?<=HOSTID=)(.*?)(?= )")
USERNAME=$(echo $grbprobe_output | grep -Po "(?<=USERNAME=)(.*?)(?= )")
CORES=$(echo $grbprobe_output | grep -Po "(?<=CORES=)(.*?)(?= )")

# Should generate a key from the gurobi website each time a new AWS instance is created
echo "Please obtain a gurobi KEY from https://portal.gurobi.com/iam/licenses/request/?type=academic"
KEY=to-be-filled

# The url can only be accessed with terminals which are connected to the university network
probe_url="https://portal.gurobi.com/keyserver?id=${KEY}&hostname=${HOSTNAME}&hostid=${HOSTID}&username=${USERNAME}&os=linux&localdate=2024-05-17&version=10&cores=${CORES}"
echo $probe_url
