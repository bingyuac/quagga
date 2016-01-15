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
class CpuContext(object):
    """
    Mock class created for compatibility purposes in order to enable
    quick switching between GPU and CPU implementations.
    """
    def __init__(self, device_id=None):
        self.device_id = device_id if device_id else 0

    def synchronize(self):
        pass

    def wait(self, *args):
        pass

    def block(self, *args):
        pass

    def add_callback(self, callback, *args, **kwargs):
        callback(*args, **kwargs)

    @staticmethod
    def callback(function):
        return function