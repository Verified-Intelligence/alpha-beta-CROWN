import os
import sys
import time
import torch
from torch.utils.cpp_extension import load, BuildExtension, CUDAExtension
from setuptools import setup

class DummyCudaClass:
    """A dummy class with error message when a CUDA function is called."""
    def __getattr__(self, attr):
        if attr == "double2float":
            # When CUDA module is not built successfully, use a workaround.
            def _f(x, d):
                print('WARNING: Missing CUDA kernels. Please enable CUDA build by setting environment variable AUTOLIRPA_ENABLE_CUDA_BUILD=1 for the correct behavior!')
                return x.float()
            return _f
        def _f(*args, **kwargs):
            raise RuntimeError(f"method {attr} not available because CUDA module was not built.")
        return _f

if __name__ == "__main__" and len(sys.argv) > 1:
    # Build and install native CUDA modules that can be directly imported later
    print('Building and installing native CUDA modules...')
    setup(
        name='auto_LiRPA_cuda_utils',
        ext_modules=[CUDAExtension('auto_LiRPA_cuda_utils', [
            'auto_LiRPA/cuda/cuda_utils.cpp', 
            'auto_LiRPA/cuda/cuda_kernels.cu'
        ])],
        cmdclass={'build_ext': BuildExtension.with_options()},
    )
    exit(0)

if torch.cuda.is_available() and os.environ.get('AUTOLIRPA_ENABLE_CUDA_BUILD', False):
    try:
        import auto_LiRPA_cuda_utils as _cuda_utils
    except:
        print('CUDA modules have not been installed')        
        try:
            print('Building native CUDA modules...')
            code_dir = os.path.dirname(os.path.abspath(__file__))
            verbose = os.environ.get('AUTOLIRPA_DEBUG_CUDA_BUILD', None) is not None
            _cuda_utils = load(
                'cuda_utils', [os.path.join(code_dir, 'cuda', 'cuda_utils.cpp'), os.path.join(code_dir, 'cuda', 'cuda_kernels.cu')], verbose=verbose)
            print('CUDA modules have been built.')
        except:
            print('CUDA module build failure. Some features will be unavailable.')
            print('Please make sure the latest CUDA toolkit is installed in your system.')
            if verbose:
                print(sys.exc_info()[2])
            else:
                print('Set environment variable AUTOLIRPA_DEBUG_CUDA_BUILD=1 to view build log.')
            _cuda_utils = DummyCudaClass() 
else:
    if os.environ.get('AUTOLIRPA_ENABLE_CUDA_BUILD', False):
        print('CUDA unavailable. Some features are disabled.')
    _cuda_utils = DummyCudaClass() 

double2float = _cuda_utils.double2float

def test_double2float():
    # Test the double2float function.
    import time
    shape = (3,4,5)

    a = torch.randn(size=shape, dtype=torch.float64, device='cuda')
    a = a.transpose(0,1)

    au = _cuda_utils.double2float(a, "up")
    ad = _cuda_utils.double2float(a, "down")

    print(a.size(), au.size(), ad.size())

    a_flatten = a.reshape(-1)
    au_flatten = au.reshape(-1)
    ad_flatten = ad.reshape(-1)

    for i in range(a_flatten.numel()):
        ai = a_flatten[i].item()
        aui = au_flatten[i].item()
        adi = ad_flatten[i].item()
        print(adi, ai, aui)
        assert adi <= ai
        assert aui >= ai
    del a, au, ad, a_flatten, au_flatten, ad_flatten

    # Performance benchmark.
    for j in [1, 4, 16, 64, 256, 1024]:
        shape = (j, 512, 1024)
        print(f'shape: {shape}')
        t = torch.randn(size=shape, dtype=torch.float64, device='cuda')

        torch.cuda.synchronize()
        start_time = time.time()
        for i in range(10):
            tt = t.float()
        torch.cuda.synchronize()
        del tt
        pytorch_time = time.time() - start_time
        print(f'pytorch rounding time: {pytorch_time:.4f}')

        torch.cuda.synchronize()
        start_time = time.time()
        for i in range(10):
            tu = _cuda_utils.double2float(t, "up")
        torch.cuda.synchronize()
        del tu
        roundup_time = time.time() - start_time
        print(f'cuda round up time: {roundup_time:.4f}')

        torch.cuda.synchronize()
        start_time = time.time()
        for i in range(10):
            td = _cuda_utils.double2float(t, "down")
        torch.cuda.synchronize()
        del td
        rounddown_time = time.time() - start_time
        print(f'cuda round down time: {rounddown_time:.4f}')

        del t


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Some tests. It's not possible to test them automatically because travis does not have CUDA.
        test_double2float()
