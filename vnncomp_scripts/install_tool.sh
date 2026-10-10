#!/bin/bash

# Installation script used for VNN-COMP. Targets Ubuntu 24.04/26.04.

TOOL_NAME=alpha-beta-CROWN
VERSION_STRING=v1
UV_ENV_DIR=${HOME}/UV_ENVS/alpha-beta-crown
if [[ -z "${VNNCOMP_PYTHON_PATH}" ]]; then
	VNNCOMP_PYTHON_PATH=${UV_ENV_DIR}/bin
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
# linux-headers-$(uname -r) is required for the NVIDIA DKMS module build;
# default-jre provides a working JVM for the CPLEX (InstallAnywhere) installer.
sudo -E DEBIAN_FRONTEND=noninteractive apt install -y sudo vim-gtk3 curl wget git cmake tmux aria2 build-essential netcat-openbsd expect dkms default-jre patchelf "linux-headers-$(uname -r)"

sudo systemctl stop cron.service chrony.service multipathd.service multipathd.socket udisks2.service packagekit.service polkit.service networkd-dispatcher.service
sudo systemctl disable cron.service chrony.service multipathd.service multipathd.socket udisks2.service packagekit.service polkit.service networkd-dispatcher.service
sudo systemctl mask cron.service chrony.service multipathd.service multipathd.socket udisks2.service packagekit.service polkit.service networkd-dispatcher.service

grep AMD /proc/cpuinfo > /dev/null && echo "export MKL_DEBUG_CPU_TYPE=5" >> ${HOME}/.profile
echo "export OMP_NUM_THREADS=1" >> ${HOME}/.profile

# Install uv (https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export PATH=${PATH}:'${HOME}'/.local/bin' >> ~/.profile
echo "alias py37=\"source ${UV_ENV_DIR}/bin/activate\"" >> ${HOME}/.profile
export PATH=${PATH}:$HOME/.local/bin

# Install NVIDIA driver
DRIVER_VERSION=610.43.02
aria2c -x 10 -s 10 -k 1M https://us.download.nvidia.com/XFree86/Linux-x86_64/$DRIVER_VERSION/NVIDIA-Linux-x86_64-$DRIVER_VERSION.run
# The nouveau driver must be unloaded/blacklisted first; otherwise the .run
# installer aborts ("The Nouveau kernel driver is currently in use ...").
sudo bash -c 'printf "blacklist nouveau\noptions nouveau modeset=0\n" > /etc/modprobe.d/blacklist-nouveau.conf'
sudo rmmod nouveau 2>/dev/null || true
sudo update-initramfs -u
chmod +x ./NVIDIA-Linux-x86_64-$DRIVER_VERSION.run
# --dkms builds the kernel module (needs linux-headers, installed above);
# --disable-nouveau lets the installer tear down nouveau on its own.
sudo ./NVIDIA-Linux-x86_64-$DRIVER_VERSION.run --silent --dkms --disable-nouveau
# Remove old driver (if already installed) and reload the new one.
sudo rmmod nvidia_uvm; sudo rmmod nvidia_drm; sudo rmmod nvidia_modeset; sudo rmmod nvidia
sudo modprobe nvidia; sudo nvidia-smi -e 0; sudo nvidia-smi -r -i 0
sudo nvidia-smi -pm 1
# Make sure GPU shows up.
nvidia-smi

# Create uv virtualenv and sync dependencies from pyproject.toml (repo root)
mkdir -p ${HOME}/UV_ENVS
uv venv --python 3.11 ${UV_ENV_DIR}
(cd ${TOOL_DIR} && VIRTUAL_ENV=${UV_ENV_DIR} uv sync --active)

# Install Gurobi 13 only for its command line tool. real solver is specified in the pyproject.toml.
aria2c -x 10 -s 10 -k 1M https://packages.gurobi.com/13.0/gurobi13.0.2_linux64.tar.gz
tar -xzf gurobi13.0.2_linux64.tar.gz
mv gurobi1302 ${HOME}/gurobi1302

# Install CPLEX
aria2c -x 10 -s 10 -k 1M "http://d.huan-zhang.com/storage/programs/cplex_studio2211.linux_x86_64.bin"
chmod +x cplex_studio2211.linux_x86_64.bin
cat > response.txt <<EOF
INSTALLER_UI=silent
LICENSE_ACCEPTED=true
EOF

sudo GLIBC_TUNABLES=glibc.rtld.execstack=2 ./cplex_studio2211.linux_x86_64.bin -f response.txt

# Build CPLEX interface
make -C ${TOOL_DIR}/complete_verifier/cuts/CPLEX_cuts/

echo "Checking python requirements (it might take a while...)"
TORCH_VERSION=$(${VNNCOMP_PYTHON_PATH}/python -c 'import torch; print(torch.__version__)')
if [[ "${TORCH_VERSION}" != 2.11.0* ]]; then
    echo "Unsupported PyTorch version: ${TORCH_VERSION}"
    echo "Installation Failure!"
    exit 1
fi


# Setup Gurobi
grbprobe_output=$(${HOME}/gurobi1302/linux64/bin/grbprobe)
echo $grbprobe_output

HOSTNAME=$(echo $grbprobe_output | grep -Po "(?<=HOSTNAME=)(.*?)(?= )")
HOSTID=$(echo $grbprobe_output | grep -Po "(?<=HOSTID=)(.*?)(?= )")
USERNAME=$(echo $grbprobe_output | grep -Po "(?<=USERNAME=)(.*?)(?= )")
CORES=$(echo $grbprobe_output | grep -Po "(?<=CORES=)(.*?)(?= )")

# Should generate a key from the gurobi website each time a new AWS instance is created
echo "Please obtain a gurobi KEY from https://portal.gurobi.com/iam/licenses/request/?type=academic"
KEY=to-be-filled

echo "Your gurobi KEY is a long string, looking like 9c4a7e52-6f1d-4a8b-9c73-2e5d1f0b8a6e"
echo 'To obtain a licence file, replace "to-be-filled" with your key in the URL below,'
echo 'and visit the URL using a campus network'
# The url can only be accessed with terminals which are connected to the university network
probe_url="https://portal.gurobi.com/keyserver?id=${KEY}&hostname=${HOSTNAME}&hostid=${HOSTID}&username=${USERNAME}&os=linux&localdate=2026-05-17&version=10&cores=${CORES}"
echo $probe_url
echo
echo "Setup incomplete - please setup gurobi using the instructions above."
