class ValidationInterruption(object):
    def __init__(self, period, validation_model):
        self.period = period
        self.validation_model = validation_model

    def interrupt(self, model):
        # TODO perform validation
        pass