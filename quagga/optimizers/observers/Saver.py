import json
import copy
import h5py
from itertools import izip
from quagga.context import Context
from quagga.blocks import ParameterContainer


class Saver(object):
    def __init__(self, model, period, definition_file_path, parameters_file_path, logger):
        self.model = model
        self.period = period
        self.definition_file_path = definition_file_path
        self.parameters_file_path = parameters_file_path
        self.logger = logger
        self.iteration = 0
        # we can use our own contexts because during Connector fprop
        # derivative matrices are filling with 0.0 in param's
        # last_usage_context to_host changes param's last_usage_context.
        # We know that parameters change only during updates of optimizer.
        # Add derivatives can't be calculated until there are jobs to be done
        # in the obtaining context, which happens to be last_usage_context.
        # That is why we are save here.
        # Parameters can be changing during callbacks calls.
        self.context = {}
        for block in model.blocks:
            if isinstance(block, ParameterContainer):
                for param in block.parameters.itervalues():
                    if param.device_id not in self.context:
                        self.context[param.device_id] = Context(param.device_id)

        self._close_h5_file = Context.callback(self._close_h5_file)
        self._write_to_h5_file = Context.callback(self._write_to_h5_file)

    def notify(self):
        if self.iteration % self.period == 0 and self.iteration != 0:
            self.logger.info('Iteration {}: start saving model ...'.format(self.iteration))
            h5_file = h5py.File(self.parameters_file_path, 'w')
            model_definition = copy.deepcopy(self.model.model_definition)
            for block, (block_name, block_definition) in izip(self.model.blocks[1:], model_definition.iteritems()):
                if isinstance(block, ParameterContainer):
                    for param_name, param_definition in block_definition.iteritems():
                        if param_name == 'type':
                            continue
                        device_id = getattr(block, param_name).device_id
                        context = self.context[device_id]
                        if param_name != 'type':
                            key = block_name + '/' + param_name
                            param_definition['init'] = ['H5pyInitializer', self.parameters_file_path, key]
                            block.send_to_host(context, param_name)
                            context.add_callback(self._write_to_h5_file, h5_file, key, block, param_name)
            contexts = self.context.values()
            context = contexts[0]
            context.wait(*contexts[1:])
            context.add_callback(self._close_h5_file, h5_file)
            with open(self.definition_file_path, 'w') as f:
                json.dump(model_definition, f)
        self.iteration += 1

    def _close_h5_file(self, h5_file):
        h5_file.close()
        self.logger.info('saved')

    def _write_to_h5_file(self, h5_file, key, block, param_name):
        h5_file[key] = block.get_npa_param(param_name)
        self.logger.info(key)