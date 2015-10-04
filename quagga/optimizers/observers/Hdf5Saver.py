import h5py
from quagga.context import Context


class Hdf5Saver(object):
    def __init__(self, param_container, period, parameters_file_path, logger):
        self.param_container = param_container
        self.period = period
        self.parameters_file_path = parameters_file_path
        self.logger = logger
        self.iteration = 0
        # we can use our own contexts because during Connector fprop
        # derivative matrices are filling with 0.0 in param's
        # last_usage_context, to_host changes param's last_usage_context.
        # We know that parameters change only during updates of optimizer.
        # Add derivatives can't be calculated until there are jobs to be done
        # in the obtaining context, which happens to be last_usage_context.
        # That is why we are save here.
        # Parameters can be changing during callbacks calls.
        self.context = {}
        for param in param_container.parameters.itervalues():
            if param.device_id not in self.context:
                self.context[param.device_id] = Context(param.device_id)
        self._save_parameters = Context.callback(self._save_parameters)

    def notify(self):
        if self.iteration % self.period == 0 and self.iteration != 0:
            self.logger.info('Iteration {}: start saving model ...'.format(self.iteration))
            h5_file = h5py.File(self.parameters_file_path, 'w')
            for param_name, param in self.param_container.parameters.iteritems():
                context = self.context[param.device_id]
                self.param_container.send_to_host(context, param_name)
            contexts = self.context.values()
            context = contexts[0]
            context.wait(*contexts[1:])
            context.add_callback(self._save_parameters, h5_file)
        self.iteration += 1

    def _save_parameters(self, h5_file):
        for param_name in self.param_container.parameters:
            h5_file[param_name] = self.param_container.get_npa_param(param_name)
            self.logger.info(param_name)
        h5_file.close()
        self.logger.info('saved')