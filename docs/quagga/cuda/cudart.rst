cudart
------

.. automodule:: quagga.cuda.cudart

.. currentmodule:: quagga.cuda.cudart

.. autofunction:: cuda_get_error_string
.. autoclass:: CudaError
.. autofunction:: check_cuda_status
.. autofunction:: cuda_get_last_error
.. autofunction:: cuda_malloc
.. autofunction:: cuda_free
.. autofunction:: cuda_malloc_host
.. autofunction:: cuda_memcpy
.. autofunction:: cuda_memcpy_async
.. autofunction:: cuda_memcpy2d_async
.. autofunction:: cuda_memcpy2d
.. autofunction:: cuda_memcpy_peer
.. autofunction:: cuda_memcpy_peer_async
.. autofunction:: cuda_mem_get_info
.. autofunction:: cuda_get_device_count
.. autofunction:: cuda_set_device
.. autofunction:: cuda_get_device
.. autofunction:: cuda_driver_get_version
.. autoclass:: CudaPointerAttributes
.. autofunction:: cuda_pointer_get_attributes
.. autofunction:: cuda_device_synchronize
.. autofunction:: cuda_stream_create
.. autofunction:: cuda_stream_destroy
.. autofunction:: cuda_stream_synchronize
.. autofunction:: cuda_stream_wait_event
.. autofunction:: cuda_stream_add_callback
.. autofunction:: cuda_event_create
.. autofunction:: cuda_event_create_with_flags
.. autofunction:: cuda_event_destroy
.. autofunction:: cuda_event_record
.. autofunction:: cuda_device_reset
.. autofunction:: cuda_set_device_flags
.. autofunction:: cuda_set_device_flags
.. autoclass:: CudaIpcMemHandleType
.. autofunction:: cuda_ipc_get_mem_handle
.. autofunction:: cuda_ipc_open_mem_handle
.. autofunction:: cuda_ipc_close_mem_handle
.. autofunction:: cuda_device_enable_peer_access
.. autofunction:: cuda_device_can_access_peer
.. autofunction:: device