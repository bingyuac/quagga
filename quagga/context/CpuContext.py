class CpuContext(object):
    def __init__(self, device_id=None):
        self.device_id = device_id

    def synchronize(self):
        pass

    def wait(self, *args):
        pass

    def block(self, *args):
        pass