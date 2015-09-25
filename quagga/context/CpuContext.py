class CpuContext(object):
    def __init__(self, device_id=None):
        self.device_id = device_id if device_id else 0

    def synchronize(self):
        pass

    def wait(self, *args):
        pass

    def block(self, *args):
        pass

    def add_callback(self, callback, *args, **kwargs):
        callback(*args, **kwargs)

    @staticmethod
    def callback(function):
        return function