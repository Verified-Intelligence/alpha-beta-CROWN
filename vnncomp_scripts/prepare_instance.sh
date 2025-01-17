#!/bin/bash

TOOL_NAME=alpha-beta-CROWN
VERSION_STRING=v1

if [[ -z "${VNNCOMP_PYTHON_PATH}" ]]; then
	VNNCOMP_PYTHON_PATH=/home/ubuntu/miniconda/envs/alpha-beta-crown/bin
fi
echo $VNNCOMP_PYTHON_PATH

# check arguments
if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi

CATEGORY=$2
ONNX_FILE=$3
VNNLIB_FILE=$4

echo "Preparing $TOOL_NAME for benchmark instance in category '$CATEGORY' with onnx file '$ONNX_FILE' and vnnlib file '$VNNLIB_FILE'"

TOOL_DIR=$(dirname $(dirname $(realpath $0)))
echo TOOL_DIR is $TOOL_DIR

export PYTHONPATH=${TOOL_DIR}
export OMP_NUM_THREADS=1

# If ABCROWN_TEST_RUN is 1, then we skip the extra steps to make the run faster, for debugging propose.
if [[ "${ABCROWN_TEST_RUN}" -ne "1" ]]; then

	# kill any zombie processes
	killall -q python
	killall -q python3
	killall -q get_cuts
	killall -q -9 python
	killall -q -9 python3
	killall -q -9 get_cuts
	sleep 3
	# Reset GPU, make sure nothing is running.
	(sudo -n rmmod nvidia_uvm; sudo -n rmmod nvidia_drm; sudo -n rmmod nvidia_modeset; sudo -n nvidia-smi -e 0; sudo -n nvidia-smi -pm 0; sudo -n nvidia-smi -r -i 0) > /dev/null
	sudo -n modprobe nvidia_uvm; sudo -n nvidia-smi -pm 1
	# Make sure GPU shows up.
	nvidia-smi

fi

if [[ "$CATEGORY" == "cgan" || "$CATEGORY" == "cgan_2023" ]]; then
	if [ -f "$ONNX_FILE.original" ]; then
		echo 'cgan models previously simplified'
	else
		echo 'cgan simplifying...'
		${VNNCOMP_PYTHON_PATH}/onnxsim "$ONNX_FILE" output.onnx
		cp $ONNX_FILE $ONNX_FILE.original
		cp output.onnx $ONNX_FILE
	fi
fi

# Warmup, using a 1 second timeout.
echo
echo "Running warmup..."
echo
temp_file=$(mktemp)
prepare_timeout=300
echo "Preparation time budget is ${prepare_timeout} seconds for $CATEGORY"
if [[ "${ABCROWN_TEST_RUN}" -ne "1" ]]; then
	timeout -k 5 ${prepare_timeout} ${VNNCOMP_PYTHON_PATH}/python ${TOOL_DIR}/complete_verifier/vnncomp_main.py "$CATEGORY" "$ONNX_FILE" "$VNNLIB_FILE" "$temp_file" 1 --PREPARE >/dev/null
	rm ${temp_file}

	# kill any remaining python processes.
	killall -q python
	killall -q python3
	killall -q get_cuts
	sleep 1
	killall -q -9 python
	killall -q -9 python3
	killall -q -9 get_cuts
	sleep 3
fi

echo "Preparation finished."

# script returns a 0 exit code if successful. If you want to skip a benchmark category you can return non-zero.
exit 0
