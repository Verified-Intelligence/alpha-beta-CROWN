#!/bin/bash

VERSION_STRING=v1
if [[ -z "${VNNCOMP_PYTHON_PATH}" ]]; then
	VNNCOMP_PYTHON_PATH=/home/ubuntu/anaconda3/envs/alpha-beta-crown/bin
fi

# check arguments
if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi

CATEGORY=$2
ONNX_FILE=$3
VNNLIB_FILE=$4
RESULTS_FILE=$5
TIMEOUT=$6

echo "Running benchmark instance in category '$CATEGORY' with onnx file '$ONNX_FILE', vnnlib file '$VNNLIB_FILE', results file $RESULTS_FILE, and timeout $TIMEOUT"

# setup environment variable for tool (doing it earlier won't be persistent with docker)"
TOOL_DIR=$(dirname $(dirname $(realpath $0)))

# Remove old results file.
rm $RESULTS_FILE

# run the tool to produce the results file
exec ${VNNCOMP_PYTHON_PATH}/python3 ${TOOL_DIR}/complete_verifier/vnncomp_main.py "$CATEGORY" "$ONNX_FILE" "$VNNLIB_FILE" "$RESULTS_FILE" "$TIMEOUT"
