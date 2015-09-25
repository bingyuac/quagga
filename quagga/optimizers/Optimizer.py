class Optimizer(object):
    def __init__(self, stopping_criterion, model):
        self.stopping_criterion = stopping_criterion
        self.model = model
        self.observers = []

    def optimize(self):
        self.model.set_training_mode()
        while not self.stopping_criterion.is_fulfilled():
            self.model.fprop()
            self.model.bprop()
            for observer in self.observers:
                observer.notify()

    def add_observer(self, observer):
        self.observers.append(observer)