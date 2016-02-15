# ----------------------------------------------------------------------------
# Copyright 2015 Grammarly, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
import h5py
from quagga.context import Context


class Hdf5Saver(object):
    def __init__(self, params, period, parameters_file_path, logger):
        self.params = params
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
        for param in params.itervalues():
            if param.device_id not in self.context:
                self.context[param.device_id] = Context(param.device_id)
        self.npa_params = {}

    def notify(self):
        if self.iteration % self.period == 0 and self.iteration != 0:
            self.logger.info('Iteration {}: start saving model ...'.format(self.iteration))
            h5_file = h5py.File(self.parameters_file_path, 'w')
            for param_name, param in self.params.iteritems():
                context = self.context[param.device_id]
                self.npa_params[param_name] = param.to_host(context)
            contexts = self.context.values()
            context = contexts[0]
            context.wait(*contexts[1:])
            context.add_callback(self._save_parameters, h5_file)
        self.iteration += 1

    def _save_parameters(self, h5_file):
        for param_name in self.params.iterkeys():
            h5_file[param_name] = self.npa_params[param_name]
            self.logger.info(param_name)
        h5_file.close()
        self.npa_params.clear()
        self.logger.info('saved')