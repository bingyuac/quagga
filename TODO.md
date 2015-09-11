# TODO


- [x] Add multi-gpu Context (http://on-demand.gputechconf.com/gtc-express/2011/presentations/cuda_webinars_multi_gpu.pdf)
- [ ] Add reduction kernels for mean value
- [ ] Add compiler functionality for more flexible code generation
- [ ] Add max margin cost function
- [ ] use device api for dropout instead of host api
- [ ] Check all np.allclose into test_Matrix tests for the right checking, sometimes there are typos 
- [ ] Fix blocks tests and unify block interface
- [ ] Add NCE block 
- [ ] Add strides support https://github.com/inducer/pycuda/blob/master/pycuda/gpuarray.py#L1105
- [ ] Follow pep8 and http://docs.openstack.org/developer/hacking/
- [ ] add order to GpuMatrix 'C' order can help speed up slicing in EmbeddingBlock
- [ ] add license
- [ ] write readme
- [ ] Add callbacks in streams for observers in order to make code truly async (https://docs.python.org/2/extending/extending.html#calling-python-functions-from-c) begin with CFUNCTYPE in ctypes, dynamically generate cuda callback functions
- [ ] give proper name for AddLastBlock, AddFirstBlock, RemoveFirstBlock, RemoveLastBlock
- [ ] add gradient clipping page 5/6 http://arxiv.org/pdf/1308.0850v5.pdf