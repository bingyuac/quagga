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
from quagga.blocks.ArgmaxBlock import ArgmaxBlock
from quagga.blocks.AttentionBlock import AttentionBlock
from quagga.blocks.ColSlicingBlock import ColSlicingBlock
from quagga.blocks.DotBlock import DotBlock
from quagga.blocks.DropoutBlock import DropoutBlock
from quagga.blocks.GaussianNoiseBlock import GaussianNoiseBlock
from quagga.blocks.GradientReversalBlock import GradientReversalBlock
from quagga.blocks.HorizontalStackBlock import HorizontalStackBlock
from quagga.blocks.InputlessLstmBlock import InputlessLstmBlock
from quagga.blocks.L2RegularizationBlock import L2RegularizationBlock
from quagga.blocks.LastSelectorBlock import LastSelectorBlock
from quagga.blocks.LstmBlock import LstmBlock
from quagga.blocks.MeanPoolingBlock import MeanPoolingBlock
from quagga.blocks.NonlinearityBlock import NonlinearityBlock
from quagga.blocks.ParameterContainer import ParameterContainer
from quagga.blocks.RepeatBlock import RepeatBlock
from quagga.blocks.RowSlicingBlock import RowSlicingBlock
from quagga.blocks.RowSlicingBlockDense import RowSlicingBlockDense
from quagga.blocks.ScheduledSamplingBlock import ScheduledSamplingBlock
from quagga.blocks.SequencerBlock import SequencerBlock
from quagga.blocks.SequentialHorizontalStackBlock import SequentialHorizontalStackBlock
from quagga.blocks.SequentialMeanPoolingBlock import SequentialMeanPoolingBlock
from quagga.blocks.SequentialSumPoolingBlock import SequentialSumPoolingBlock
from quagga.blocks.SigmoidCeBlock import SigmoidCeBlock
from quagga.blocks.SoftmaxBlock import SoftmaxBlock
from quagga.blocks.SoftmaxCeBlock import SoftmaxCeBlock
from quagga.blocks.VerticalStackBlock import VerticalStackBlock
from quagga.blocks.SseBlock import SseBlock