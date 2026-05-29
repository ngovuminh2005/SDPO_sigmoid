# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import pytest
import torch.nn as nn

from verl.workers.config.optimizer import FSDPOptimizerConfig, build_optimizer


class TestFSDPOptimizerConfigCPU:
    def test_default_configuration(self):
        config = FSDPOptimizerConfig(lr=0.1)
        assert config.min_lr_ratio is None
        assert config.lr_scheduler_type == "constant"
        assert config.num_cycles == 0.5

    @pytest.mark.parametrize("lr_scheduler_type", ["constant", "cosine"])
    def test_valid_lr_scheduler_types(self, lr_scheduler_type):
        config = FSDPOptimizerConfig(lr_scheduler_type=lr_scheduler_type, lr=0.1)
        assert config.lr_scheduler_type == lr_scheduler_type

    @pytest.mark.parametrize("warmup_style", ["constant", "cosine"])
    def test_valid_warmup_style_types(self, warmup_style):
        config = FSDPOptimizerConfig(warmup_style=warmup_style, lr=0.1)
        assert config.lr_scheduler_type == warmup_style

    def test_invalid_lr_scheduler_type(self):
        with pytest.raises((ValueError, AssertionError)):
            FSDPOptimizerConfig(lr_scheduler_type="invalid_style", lr=0.1)

    def test_invalid_warmup_style_type(self):
        with pytest.raises((ValueError, AssertionError)):
            FSDPOptimizerConfig(warmup_style="invalid_style", lr=0.1)

    @pytest.mark.parametrize("num_cycles", [0.1, 1.0, 2.5])
    def test_num_cycles_configuration(self, num_cycles):
        config = FSDPOptimizerConfig(num_cycles=num_cycles, lr=0.1)
        assert config.num_cycles == num_cycles

    def test_muon_auxadam_requires_module(self):
        cfg = FSDPOptimizerConfig(lr=1e-3, optimizer="SingleDeviceMuonWithAuxAdam", optimizer_impl="torch.optim")
        m = nn.Linear(8, 4)
        with pytest.raises(ValueError, match="requires `module`"):
            build_optimizer(m.parameters(), cfg)

    def test_muon_auxadam_builds_with_module(self):
        cfg = FSDPOptimizerConfig(
            lr=1e-3,
            optimizer="SingleDeviceMuonWithAuxAdam",
            optimizer_impl="torch.optim",
            weight_decay=0.01,
            muon_lr=2e-3,
            adam_aux_lr=1e-3,
        )
        m = nn.Sequential(nn.Linear(8, 4), nn.LayerNorm(4))
        opt = build_optimizer(m.parameters(), cfg, module=m)
        assert opt is not None
        assert len(opt.param_groups) >= 1

    def test_memory_muon_auxadam_builds_with_module(self):
        cfg = FSDPOptimizerConfig(
            lr=1e-3,
            optimizer="SingleDeviceMemoryMuonWithAuxAdam",
            optimizer_impl="torch.optim",
            weight_decay=0.01,
            muon_lr=2e-3,
            adam_aux_lr=1e-3,
            muon_nesterov=True,
            muon_ns_steps=5,
            memory_num_centroids=8,
            memory_centroid_dim=32,
            memory_lambda=0.02,
            memory_init_seed=123,
            memory_ema_decay=0.95,
            memory_ema_eps=1e-8,
            memory_step_update=4,
            memory_ema_ws_steps=7,
        )
        m = nn.Sequential(nn.Linear(8, 4), nn.LayerNorm(4))
        opt = build_optimizer(m.parameters(), cfg, module=m)
        assert opt is not None
        assert opt.ema_ws_steps == 7
        assert len(opt.param_groups) >= 1
