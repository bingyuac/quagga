class CpuContext(object):
    def __init__(self, device_id=None):
        self.device_id = device_id
        pass

    def synchronize(self):
        pass

    def depend_on(self, *args):
        pass

    def block(self, *args):
        pass