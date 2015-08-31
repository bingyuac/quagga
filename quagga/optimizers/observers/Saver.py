class Saver(object):
    def __init__(self, model, period, definition_file_path, parameters_file_path, logger):
        self.model = model
        self.period = period
        self.definition_file_path = definition_file_path
        self.parameters_file_path = parameters_file_path
        self.logger = logger
        self.iteration = 0

    def notify(self):
        if self.iteration % self.period == 0 and self.iteration != 0:
            self.logger.info('Iteration {}: start saving model ...'.format(self.iteration))
            self.model.save(self.definition_file_path, self.parameters_file_path)
            self.logger.info('saved')
        self.iteration += 1