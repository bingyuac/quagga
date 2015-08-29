class Snapshotter(object):
    def __init__(self, model, period, snapshot_file_name, logger):
        self.model = model
        self.period = period
        self.snapshot_file_name = snapshot_file_name
        self.logger = logger
        self.iteration = 0

    def notify(self):
        if self.iteration % self.period == 0 and self.iteration != 0:
            self.model.save(self.snapshot_file_name)
        self.iteration += 1