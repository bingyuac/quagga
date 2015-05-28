import numpy as np
from matrix import GpuMatrix, GpuMatrixContext
from multiprocessing import Process, Pipe


def consumer(conn):
    a, c = conn.recv()
    a.scale(c, 2)
    conn.send((a, c))


def producer(conn, a, c):
    print 'k'
    conn.send((a, c))
    a, c = conn.recv()
    c.synchronize()
    print a.to_host()


producer_conn, consumer_conn = Pipe()

p1 = Process(target=consumer, args=(producer_conn, ))
a = GpuMatrix.from_npa(np.array([[1, 2], [3, 4]]))
c = GpuMatrixContext()
p2 = Process(target=producer, args=(consumer_conn, a, c))
p1.start()
p2.start()
