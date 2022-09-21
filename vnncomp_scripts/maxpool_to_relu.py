""" https://github.com/stanleybak/vnncomp2022/issues/2#issuecomment-1159092312 
Need to install DNNV first: 
pip install git+https://github.com/dlshriver/DNNV.git@develop
"""

import time

import sys

from dnnv.nn import parse
from dnnv.nn.transformers.simplifiers import (simplify, ReluifyMaxPool)
from pathlib import Path

def main():
    """main entry point"""

    assert len(sys.argv) == 2, "expected 1 arguments: [input .onnx filename]"

    onnx_filename = sys.argv[1]
     
    op_graph = parse(Path(onnx_filename))

    print("starting...")
    t = time.perf_counter()
    simplified_op_graph1 = simplify(op_graph, ReluifyMaxPool(op_graph))
    diff = time.perf_counter() - t
    print(f"simplify runtime: {diff}")
    
    print("exporting...")
    t = time.perf_counter()
    simplified_op_graph1.export_onnx('output.onnx')
    diff = time.perf_counter() - t
    print(f"export runtime: {diff}")

if __name__ == "__main__":
    main()
