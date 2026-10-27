#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
"""Backport vLLM PP + MTP runtime support.

The local Eagle/MTP drafter returns the draft tokens that belong to the model
output being processed. With PP batch_queue, EngineCore schedules a newer batch
before consuming the older output, so updating ``request.spec_token_ids`` from
``post_step`` observes live Request state from the newer schedule step.
"""

from __future__ import annotations

import copy
from functools import wraps

from vllm.logger import logger

_PATCHED = False


def _patch_model_config_validation() -> None:
    from typing import get_args

    from vllm.config.model import ModelConfig
    from vllm.config.speculative import MTPModelTypes

    original_verify = ModelConfig.verify_with_parallel_config
    if getattr(original_verify, "_vllm_ascend_pp_mtp_patched", False):
        return

    mtp_model_types = set(get_args(MTPModelTypes))

    @wraps(original_verify)
    def _patched_verify_with_parallel_config(self, parallel_config):
        hf_config = getattr(self, "hf_config", None)
        model_type = getattr(hf_config, "model_type", None)
        is_eagle_drafter = (model_type == "eagle" or model_type == "speculators") and any(
            arch.startswith("Eagle") or arch.endswith("Eagle3") for arch in getattr(self, "architectures", ())
        )
        is_mtp_drafter = model_type in mtp_model_types
        if (
            getattr(self, "runner", None) == "draft"
            and (is_eagle_drafter or is_mtp_drafter)
            and getattr(parallel_config, "pipeline_parallel_size", 1) > 1
        ):
            # Local Eagle/MTP drafters are loaded on the last PP stage rather
            # than partitioned across all PP stages. Keep normal target-model
            # validation intact, but validate these draft models as PP=1.
            logger.warning(
                "Validating local Eagle/MTP drafter with pipeline_parallel_size=1 "
                "because it is loaded locally on the last pipeline stage."
            )
            patched_config = copy.copy(parallel_config)
            patched_config.pipeline_parallel_size = 1
            return original_verify(self, patched_config)
        return original_verify(self, parallel_config)

    _patched_verify_with_parallel_config._vllm_ascend_pp_mtp_patched = True  # type: ignore[attr-defined]
    ModelConfig.verify_with_parallel_config = _patched_verify_with_parallel_config



def _patch_max_concurrent_batches() -> None:
    """Serialize PP+MTP scheduler batches for V3 correctness.

    Upstream PP uses a batch queue of depth PP.  With a local MTP proposer,
    EngineCore can schedule the next batch before the previous draft token ids
    have returned from the last pipeline stage.  Stage A then embeds -1
    placeholders and corrupts target state.  The V3 bridge intentionally runs
    one scheduler batch at a time until upstream owns this synchronization.
    """
    from vllm.config import VllmConfig

    original = VllmConfig.max_concurrent_batches
    if getattr(original, "_v3_pp_mtp_patched", False):
        return

    def patched(self):
        parallel = getattr(self.parallel_config, "pipeline_parallel_size", 1)
        speculative = getattr(self, "speculative_config", None)
        method = str(getattr(speculative, "method", "") or "").lower()
        additional = getattr(self, "additional_config", None) or {}
        v3 = additional.get("v3_verdict_adaptive", {}) if isinstance(additional, dict) else {}
        scheduler_fix = isinstance(v3, dict) and bool(v3.get("fix_pp_mtp", v3.get("enabled", False)))
        serialized = (
            isinstance(v3, dict)
            and scheduler_fix
            and bool(v3.get("serialize_spec", True))
        )
        pipelined = isinstance(v3, dict) and bool(v3.get("pipeline_spec", False))
        if serialized and parallel > 1 and "mtp" in method:
            return max(2, parallel) if pipelined else 1
        return original.fget(self)

    patched._v3_pp_mtp_patched = True  # type: ignore[attr-defined]
    VllmConfig.max_concurrent_batches = property(patched)


def _patch_engine_core_spec_pp_batch_queue() -> None:
    """Pipeline PP+MTP without scheduling stage A on stale draft placeholders.

    Upstream's PP batch queue schedules a second scheduler output before the
    first stage-B MTP proposer has returned draft token ids.  The V3 pipeline
    keeps at most one completed-but-unprocessed output in the queue.  After
    processing that output, it imports the just-returned draft ids and only then
    schedules the next PP micro-batch.  Thus stage A/B execution remains pipelined
    across EngineCore calls, while draft consumption always observes a fence.
    """
    from concurrent.futures import Future
    from typing import cast

    from vllm.v1.engine.core import EngineCore
    from vllm.v1.outputs import ModelRunnerOutput

    original = EngineCore.step_with_batch_queue
    if getattr(original, "_v3_pp_mtp_patched", False):
        return

    def patched(self):
        additional = getattr(self.vllm_config, "additional_config", None) or {}
        v3 = additional.get("v3_verdict_adaptive", {}) if isinstance(additional, dict) else {}
        scheduler_fix = isinstance(v3, dict) and bool(v3.get("fix_pp_mtp", v3.get("enabled", False)))
        serialized = (
            isinstance(v3, dict)
            and scheduler_fix
            and bool(v3.get("serialize_spec", True))
        )
        parallel = getattr(self.vllm_config.parallel_config, "pipeline_parallel_size", 1)
        speculative = getattr(self.vllm_config, "speculative_config", None)
        method = str(getattr(speculative, "method", "") or "").lower()
        if not serialized or parallel <= 1 or "mtp" not in method:
            return original(self)

        batch_queue = self.batch_queue
        assert batch_queue is not None

        def schedule_and_execute():
            scheduler_output = self.scheduler.schedule()
            with self.log_error_detail(scheduler_output):
                exec_future = self.model_executor.execute_model(scheduler_output, non_block=True)
            model_executed = scheduler_output.total_num_scheduled_tokens > 0
            if self.is_ec_consumer:
                model_executed = scheduler_output.total_num_scheduled_tokens > 0
            if self.is_pooling_model or not model_executed:
                future = cast(Future[ModelRunnerOutput], exec_future)
            elif scheduler_output.pending_structured_output_tokens:
                raise ValueError(
                    "V3 pipelined PP+MTP currently requires structured-output "
                    "deferred sampling to be disabled"
                )
            else:
                grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
                future = self.model_executor.sample_tokens(grammar_output, non_block=True)
            return future, scheduler_output, exec_future, model_executed

        # Cold call: start the first PP micro-batch but do not schedule a second
        # output before its draft ids exist.
        if not batch_queue:
            if not self.scheduler.has_requests():
                return None, False
            future, scheduler_output, exec_future, model_executed = schedule_and_execute()
            batch_queue.appendleft((future, scheduler_output, exec_future))
            return None, model_executed

        # Steady state: consume one completed output, import its draft ids, then
        # schedule exactly one new output.  The newly scheduled future remains in
        # flight while EngineCore returns the previous committed result.
        future, scheduler_output, exec_future = batch_queue.pop()
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            model_output = future.result()
            if model_output is None:
                exec_future.result()
                raise RuntimeError("unexpected error")
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(scheduler_output, model_output)

        draft_token_ids = self.model_executor.take_draft_token_ids()
        if draft_token_ids is not None:
            self.scheduler.update_draft_token_ids(draft_token_ids)

        if self.scheduler.has_requests():
            next_future, next_output, next_exec_future, next_executed = schedule_and_execute()
            if next_future is not None:
                batch_queue.appendleft((next_future, next_output, next_exec_future))
        return engine_core_outputs, True

    patched._v3_pp_mtp_patched = True  # type: ignore[attr-defined]
    EngineCore.step_with_batch_queue = patched

def _apply_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True
    _patch_model_config_validation()
    _patch_max_concurrent_batches()
    _patch_engine_core_spec_pp_batch_queue()


_apply_patch()
