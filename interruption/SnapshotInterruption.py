class SnapshotInterruption(object):
    def __init__(self, period, snapshot_dir):
        """

        :param period: number of optimizers iteration after which `interrupt` method will be called
        :param snapshot_dir:
        """
        self.period = period
        self.snapshot_dir = snapshot_dir

    def interrupt(self, model):
        # TODO save network
        pass