# TODO


- [x] Add multi-gpu Context (http://on-demand.gputechconf.com/gtc-express/2011/presentations/cuda_webinars_multi_gpu.pdf)
- [ ] Add reduction kernels for mean and sum along the axis
- [ ] Add compiler functionality for more flexible code generation
- [ ] Add max margin cost function
- [ ] use device api for dropout instead of host api
- [ ] Add NCE block 
- [ ] Add strides support https://github.com/inducer/pycuda/blob/master/pycuda/gpuarray.py#L1105
- [ ] Follow pep8 and http://docs.openstack.org/developer/hacking/
- [ ] add order to GpuMatrix 'C' order can help speed up slicing in EmbeddingBlock
- [ ] add license
- [ ] write readme
- [ ] give proper name for AddLastBlock, AddFirstBlock, RemoveFirstBlock, RemoveLastBlock
- [ ] add gradient clipping page 5/6 http://arxiv.org/pdf/1308.0850v5.pdf
- [ ] Review all matrices that go into wait_matrices, it can cause a lot of nasty bugs, that hard to reproduce and catch 