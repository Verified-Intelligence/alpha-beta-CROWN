#!/bin/bash

TOOL_NAME=alpha-beta-CROWN
VERSION_STRING=v1
if [[ -z "${VNNCOMP_PYTHON_PATH}" ]]; then
	VNNCOMP_PYTHON_PATH=/home/ubuntu/anaconda3/envs/alpha-beta-crown/bin
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

# kill any zombie processes
killall -q python
killall -q python3
killall -q -9 python
killall -q -9 python3
# set GPU to persistent mode with fixed speed for less randomness.
sudo nvidia-smi -pm 1 > /dev/null; sudo nvidia-smi -ac 877,1530 > /dev/null

if [ "$CATEGORY" = "nn4sys" ]; then
	# For NN4sys model, we convert it here because the model is big.
	${VNNCOMP_PYTHON_PATH}/python ${TOOL_DIR}/complete_verifier/convert_nn4sys_model.py ${ONNX_FILE}
fi
# Other models will be converted on the fly.


# Warmup, using a 1 second timeout.
echo
echo "Running warmup..."
echo
temp_file=$(mktemp)
timeout -k 5 30 ${VNNCOMP_PYTHON_PATH}/python ${TOOL_DIR}/complete_verifier/vnncomp_main.py "$CATEGORY" "$ONNX_FILE" "$VNNLIB_FILE" "$temp_file" 1 > /dev/null
rm ${temp_file}

# kill any remaining python processes.
killall -q python
killall -q python3
killall -q -9 python
killall -q -9 python3

# script returns a 0 exit code if successful. If you want to skip a benchmark category you can return non-zero.
exit 0
