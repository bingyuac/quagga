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
import quagga
from quagga.context import CpuContext
from quagga.context import GpuContext


def __get_context_class():
    if quagga.processor_type == 'cpu':
        return CpuContext
    elif quagga.processor_type == 'gpu':
        return GpuContext
    else:
        raise ValueError(u'Processor type: {} is undefined'.
                         format(quagga.processor_type))


def Context(device_id=None):
    """
    Creates an instance of CpuContext or GpuContext classes. Global
    variable ``processor_type`` defines which one of the two classes will be
    used.

    Parameters
    ----------
    device_id : int
        Defines with which device the computational context will be associated

    Returns
    -------
    instance of :class:`~quagga.context.CpuContext` or \
    :class:`~quagga.context.GpuContext` class.
    """
    return __get_context_class()(device_id)