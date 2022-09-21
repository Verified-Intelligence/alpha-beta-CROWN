import onnx2pytorch
from utils import load_onnx, get_test_acc

def load_model_onnx_new(path, compute_test_acc=False):#, force_convert=False, return_onnx2pytorch_model=False):
    onnx_model = load_onnx(path)
    onnx_input_dims = onnx_model.graph.input[0].type.tensor_type.shape.dim
    onnx_shape = (-1,) + tuple(d.dim_value for d in onnx_input_dims[1:])
    pytorch_model = onnx2pytorch.ConvertModel(onnx_model, experimental=True)
    
    # TODO post-processing and optimizing the converted model

    if compute_test_acc:
        get_test_acc(pytorch_model, onnx_shape)
    
    return pytorch_model, onnx_shape
    

