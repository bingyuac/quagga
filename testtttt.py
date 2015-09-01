import numpy as np
from quagga.cuda import cudart
from quagga.matrix import GpuMatrix
from quagga.context import GpuContext

# cudart.cuda_set_device(1)
# print cudart.cuda_device_can_access_peer(1, 0)
# cudart.cuda_set_device(0)
# print cudart.cuda_device_can_access_peer(0, 0)
#
# cudart.cuda_set_device(1)
# cudart.cuda_device_enable_peer_access(1)
# cudart.cuda_set_device(1)
# cudart.cuda_device_enable_peer_access(1)
#
# print cudart.cuda_device_can_access_peer(0, 0)
# print cudart.cuda_device_can_access_peer(0, 0)


a = GpuMatrix.from_npa(np.zeros((4096, 1024), dtype=np.float32), device_id=0)
b = GpuMatrix.from_npa(np.ones((4096, 1024), dtype=np.float32), device_id=0)
c = GpuMatrix.from_npa(np.zeros((4096, 1024), dtype=np.float32), device_id=1)
d = np.ones((4096, 1024), dtype=np.float32)

context = GpuContext(1)
a.assign_add(context, c, b)
import time
time.sleep(2)
print a.to_host()
print np.allclose(a.to_host(), d)



# cudart.cuda_set_device(1)
#
#
#
# c = GpuMatrix.from_npa(np.zeros((4096, 1024), dtype=np.float32), device_id=0)
# d = np.zeros((4096, 1024), dtype=np.float32)
#
#
#
# for i in xrange(4):
#     context.synchronize()
#     c.to_device(context, d)
#     context.synchronize()
#     a.copy(context, b)
#     context.synchronize()
#     b.copy(context, a)
# print np.allclose(c.to_host(), d)
