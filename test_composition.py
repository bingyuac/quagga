# class A(object):
#     def __init__(self, val):
#         self.val = val
#
#     def f(self):
#         print self.val + 1
#
#
# class B(object):
#     def __init__(self, a):
#         self.a = a
#         self.update = False
#
#     def __getattr__(self, name):
#         attribute = getattr(self.a, name)
#         if hasattr(attribute, '__call__'):
#             def update_tracking_attribute(*args, **kwargs):
#                 self.updated = True
#                 return getattr(self.a, name)(*args, **kwargs)
#             setattr(self, name, update_tracking_attribute)
#         else:
#             def fget(self):
#                 self.updated = True
#                 print 'dashhj'
#                 return getattr(self.a, name)
#
#             def fset(self, value):
#                 print 'das'
#                 self.updated = True
#                 setattr(self.a, name, value)
#             setattr(B, name, property(fget, fset))
#
#         return getattr(self, name)
#
# a = A(4)
# b = B(a)
#
# print b.val
# b.val = 4
# print b.val + 4
# b.f()
#
# b.a = A(100)
# print b.val
# b.f()
# ============================================================================



# import ctypes
# from quagga.cuda import cudart
#
#
# n = 10
# elem_size = ctypes.sizeof(ctypes.c_float)
# nbytes = n * elem_size
# data = cudart.cuda_malloc(nbytes, ctypes.c_float)
#
# handle = cudart.CudaIpcMemHandleType()
# cudart.cuda_ipc_get_mem_handle(handle, data)
# cudart.cuda_free(data)

# ============================================================================


# import numpy as np
# from multiprocessing import Process, Pipe
#
#
# def consumer(pipe):
#     from quagga.cuda import cudart
#     from quagga.matrix import Matrix
#     while True:
#         print 'aaaa'
#         q = pipe.recv()
#         print 'bbbbb'
#         if q == 42:
#             break
#         w = q.to_host()
#         print w
#         pipe.send('I received give me the next')
#
#
# def producer(pipe):
#     from quagga.cuda import cudart
#     from quagga.matrix import Matrix
#     print 'd'
#     m = Matrix.from_npa(np.array([[1, 2],
#                                   [3, 4]]))
#     print 'fg'
#     pipe.send(m)
#     print 'sdfdsf'
#     print pipe.recv()
#     pipe.send(42)
#
#
# if __name__ == '__main__':
#     producer_end, consumer_end = Pipe()
#     proc = Process(target=consumer, args=(consumer_end, ))
#     proc.start()
#     proc = Process(target=producer, args=(producer_end, ))
#     proc.start()


# ============================================================================

import ctypes
import numpy as np
from multiprocessing import Process, Pipe


def consumer(pipe):
    q = pipe.recv()
    while True:
        print 'aaaa'
        u = pipe.recv()
        print 'bbbbb'
        if type(q) == int and q == 42:
            break

        # w = ctypes.c_void_p()
        # w.value = q
        # print w.value
        #
        # q = ctypes.cast(w, ctypes.POINTER(ctypes.c_int))
        # print q[0], q[1], q[2]
        # q = ctypes.cast(q, ctypes.POINTER(ctypes.c_int*4))
        #
        # w = np.ndarray(shape=(2, 2), dtype=np.int32, buffer=q.contents, order='F')
        # print w

        print q
        pipe.send('I received give me the next')


def producer(pipe):
    print 'd'
    m = np.array([[1, 2], [3, 4]], dtype=np.int32, order='F')
    print 'fg'

    # i_p = m.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    # void_p = ctypes.cast(i_p, ctypes.c_void_p).value
    # print void_p
    #
    # pipe.send(void_p)
    pipe.send(m)
    m[0,0]= 100
    pipe.send('sdfsd')

    print 'sdfdsf'
    print pipe.recv()
    pipe.send(42)


if __name__ == '__main__':
    producer_end, consumer_end = Pipe()
    proc = Process(target=consumer, args=(consumer_end, ))
    proc.start()
    proc = Process(target=producer, args=(producer_end, ))
    proc.start()


