# Running on my laptop

When running cuda_async_demo.py on my own laptop, I saw a lot of errors about missing CUDA libraries because Arch Linux has newer versions, for example:

```
OSError: libcudart.so.12: cannot open shared object file: No such file or directory
```

and

```
ImportError: libcudnn.so.9: cannot open shared object file: No such file or directory
```

```
ImportError: libcusparseLt.so.0: cannot open shared object file: No such file or directory
```

and more.

Run this command to install the packages in the python environment:

```
uv pip install \
    nvidia-cuda-runtime-cu12 \
    nvidia-cublas-cu12 \
    nvidia-cufft-cu12 \
    nvidia-curand-cu12 \
    nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12 \
    nvidia-cudnn-cu12 \
    nvidia-cuda-cupti-cu12 \
    nvidia-nvjitlink-cu12 \
    nvidia-nvtx-cu12 \
    nvidia-cufile-cu12 \
    nvidia-cusparselt-cu12 \
    nvidia-nccl-cu12 \
    nvidia-nvshmem-cu12
```
