"""vLLM KV connector variants used by pipelines_v2 capture."""

from __future__ import annotations

import os
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (
    ExampleHiddenStatesConnector,
    ExampleHiddenStatesConnectorMetadata,
)
from vllm.v1.core.sched.output import SchedulerOutput


class PipelinesV2HiddenStatesConnector(ExampleHiddenStatesConnector):
    """Hidden-state connector that also retains computed decode-token rows.

    The upstream ``ExampleHiddenStatesConnector`` writes only
    ``prompt_token_ids`` rows, even when a request continues through decode.
    For generated-token capture we need the KV-cache rows that have actually
    been computed so far: prompt rows during prefill, then prompt plus generated
    context rows during decode.

    vLLM samples a token from the previous token's hidden state. Therefore the
    final sampled output token may not itself have a hidden-state row unless a
    later decode step feeds it back into the model. This connector captures the
    computed token rows available in vLLM's cache; callers should derive the
    generated section from the saved tensor length, not from output length alone.
    """

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> ExampleHiddenStatesConnectorMetadata:
        meta = ExampleHiddenStatesConnectorMetadata()

        for new_req in scheduler_output.scheduled_new_reqs:
            prompt_token_ids = list(new_req.prompt_token_ids or [])
            scheduled_count = int(scheduler_output.num_scheduled_tokens.get(new_req.req_id, len(prompt_token_ids)))
            capture_count = min(
                len(prompt_token_ids),
                self._available_token_count(
                    block_ids=list(new_req.block_ids[0]),
                    requested_count=int(new_req.num_computed_tokens) + scheduled_count,
                ),
            )
            filename = os.path.join(self._storage_path, f"{new_req.req_id}.safetensors")
            meta.add_request(
                new_req.req_id,
                filename=filename,
                token_ids=prompt_token_ids[:capture_count],
                block_ids=list(new_req.block_ids[0]),
                block_size=self._block_size,
            )
            self._request_filenames[new_req.req_id] = filename
            self._active_requests[new_req.req_id] = new_req
            self._req_blocks[new_req.req_id] = list(new_req.block_ids[0])

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            if req_id not in self._active_requests:
                continue

            req_block_ids = self._req_blocks[req_id]
            new_block_ids = cached_reqs.new_block_ids[i]
            if new_block_ids is not None:
                req_block_ids.extend(list(new_block_ids[0]))

            scheduled_count = int(scheduler_output.num_scheduled_tokens.get(req_id, 0))
            requested_count = int(cached_reqs.num_computed_tokens[i]) + scheduled_count
            capture_count = self._available_token_count(
                block_ids=req_block_ids,
                requested_count=requested_count,
            )
            token_ids = self._token_ids_for_capture_count(
                prompt_token_ids=list(self._active_requests[req_id].prompt_token_ids or []),
                all_token_ids=list(cached_reqs.all_token_ids.get(req_id, ())),
                capture_count=capture_count,
            )
            filename = os.path.join(self._storage_path, f"{req_id}.safetensors")
            meta.add_request(
                req_id=req_id,
                filename=filename,
                token_ids=token_ids,
                block_ids=req_block_ids,
                block_size=self._block_size,
                new_req=False,
            )

        return meta

    def request_finished(self, request: Any, block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        req_id = request.request_id
        req_filename = self._request_filenames.pop(req_id, None)
        _ = self._active_requests.pop(req_id, None)
        _ = self._req_blocks.pop(req_id, None)

        return False, {"hidden_states_path": req_filename}

    def _available_token_count(self, *, block_ids: list[int], requested_count: int) -> int:
        available_slots = len(block_ids) * int(self._block_size)
        return max(0, min(int(requested_count), available_slots))

    @staticmethod
    def _token_ids_for_capture_count(
        *,
        prompt_token_ids: list[int],
        all_token_ids: list[int],
        capture_count: int,
    ) -> list[int]:
        if len(all_token_ids) >= int(capture_count):
            return [int(token) for token in all_token_ids[:capture_count]]
        tokens = [int(token) for token in prompt_token_ids[:capture_count]]
        if len(tokens) < int(capture_count):
            tokens.extend([-1] * (int(capture_count) - len(tokens)))
        return tokens
