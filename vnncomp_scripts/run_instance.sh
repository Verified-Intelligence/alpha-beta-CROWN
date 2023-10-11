#!/bin/bash

VERSION_STRING=v1
if [[ -z "${VNNCOMP_PYTHON_PATH}" ]]; then
	VNNCOMP_PYTHON_PATH=/home/ubuntu/miniconda/envs/alpha-beta-crown/bin
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

echo
echo "Running benchmark instance in category '$CATEGORY' with onnx file '$ONNX_FILE', vnnlib file '$VNNLIB_FILE', results file $RESULTS_FILE, and timeout $TIMEOUT"

# setup environment variable for tool (doing it earlier won't be persistent with docker)"
TOOL_DIR=$(dirname $(dirname $(realpath $0)))

# Remove old results file.
rm $RESULTS_FILE

export PYTHONPATH=${TOOL_DIR}
export OMP_NUM_THREADS=1

# run the tool to produce the results file
${VNNCOMP_PYTHON_PATH}/python3 ${TOOL_DIR}/complete_verifier/vnncomp_main.py "$CATEGORY" "$ONNX_FILE" "$VNNLIB_FILE" "$RESULTS_FILE" "$TIMEOUT"
EXIT_CODE=$?
echo "exit code: ${EXIT_CODE}"

# retry with crown to save memory if needed
if [ 0 != ${EXIT_CODE} ]; then
        RESULT=$(head -n 1 "$RESULTS_FILE")
        # remove whitespace
        RESULT_STR=${RESULT//[[:space:]]/}
        if [[ ${RESULT_STR} == "" ]]; then
                ${VNNCOMP_PYTHON_PATH}/python3 ${TOOL_DIR}/complete_verifier/vnncomp_main.py "$CATEGORY" "$ONNX_FILE" "$VNNLIB_FILE" "$RESULTS_FILE" "$TIMEOUT" --TRY_CROWN
        fi
        # we return normally
        exit 0
fi
