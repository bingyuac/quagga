class SgdOptimizer(object):
    def __init__(self, learning_rate, model):
        self.learning_rate = learning_rate
        self.model = model

        params = self.model.params
        grads = self.model.grads

    def optimize(self):
        while True:
            self.update()

    def update(self):
        pass

    def add_validation(self):
        pass