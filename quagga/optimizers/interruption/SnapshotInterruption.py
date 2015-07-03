class SnapshotInterruption(object):
    def __init__(self, period, snapshot_file_name):
        """

        :param period: number of optimizers iteration after which `interrupt` method will be called
        :param snapshot_file_name:
        """
        self.period = period
        self.snapshot_dir = snapshot_file_name

    def interrupt(self, model):
        model.save(self.snapshot_dir)