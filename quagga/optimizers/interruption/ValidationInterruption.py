import logging
import numpy as np


class ValidationInterruption(object):
    def __init__(self, period, fprop_number, log_file_name, validation_model):
        self.period = period
        self.validation_model = validation_model
        self.fprop_number = fprop_number
        self.current_loss = None
        self.logger = ValidationInterruption.get_logger(log_file_name)
        self.iter = 0

    @staticmethod
    def get_logger(log_file_name):
        logger = logging.getLogger(log_file_name)
        handler = logging.FileHandler(log_file_name, 'a', encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', '%d-%m-%Y %H:%M:%S'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def interrupt(self, model):
        self.validation_model.set_testing_mode()
        loss = np.empty((self.fprop_number, ))
        for i in xrange(self.fprop_number):
            self.validation_model.fprop()
            loss[i] = self.validation_model.loss
        self.current_loss = np.mean(loss)
        self.logger.info('iter_num: {} {:1.10f}'.format(self.iter, self.current_loss))
        self.iter += self.period
        model.set_training_mode()