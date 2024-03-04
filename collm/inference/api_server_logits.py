import argparse
import json
from typing import AsyncGenerator, Optional

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

# Monkey Patching vllm Sampler to separate deferral probability
# from the rest of the sampling logic.
from vllm.model_executor.layers.sampler import (
    _SAMPLING_EPS,
    InputMetadata,
    Sampler,
    SamplerOutput,
    _apply_penalties,
    _apply_top_p_top_k,
    _build_sampler_output,
    _get_logits,
    _get_logprobs,
    _get_output_tokens,
    _get_penalties,
    _get_temperatures,
    _get_top_p_top_k,
    _prune_hidden_states,
    _sample,
)
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid


def forward(
    self,
    embedding: torch.Tensor,
    hidden_states: torch.Tensor,
    input_metadata: InputMetadata,
    embedding_bias: Optional[torch.Tensor] = None,
) -> SamplerOutput:
    # Get the hidden states that we use for sampling.
    hidden_states = _prune_hidden_states(hidden_states, input_metadata)

    # Get the logits for the next tokens.
    logits = _get_logits(hidden_states, embedding, embedding_bias, self.vocab_size)
    _final_logits = logits.clone()
    print(logits)

    # Apply presence and frequency penalties.
    output_tokens = _get_output_tokens(input_metadata)
    assert len(output_tokens) == logits.shape[0]
    presence_penalties, frequency_penalties = _get_penalties(input_metadata)
    assert len(presence_penalties) == logits.shape[0]
    assert len(frequency_penalties) == logits.shape[0]
    logits = _apply_penalties(logits, output_tokens, presence_penalties, frequency_penalties)

    # Apply temperature scaling.
    temperatures = _get_temperatures(input_metadata)
    assert len(temperatures) == logits.shape[0]
    if any(t != 1.0 for t in temperatures):
        t = torch.tensor(temperatures, dtype=logits.dtype, device=logits.device)
        # Use in-place division to avoid creating a new tensor.
        logits.div_(t.unsqueeze(dim=1))

    # Apply top-p and top-k truncation.
    top_ps, top_ks = _get_top_p_top_k(input_metadata, self.vocab_size)
    assert len(top_ps) == len(top_ks) == logits.shape[0]
    do_top_p = any(p < 1.0 - _SAMPLING_EPS for p in top_ps)
    do_top_k = any(k != self.vocab_size for k in top_ks)
    if do_top_p or do_top_k:
        logits = _apply_top_p_top_k(logits, top_ps, top_ks)

    # We use float32 for probabilities and log probabilities.
    # Compute the probabilities.
    probs = torch.softmax(logits, dim=-1, dtype=torch.float)
    # Compute the log probabilities.
    # Use log_softmax to ensure numerical stability.
    logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

    # Sample the next tokens.
    sample_results = _sample(probs, logprobs, input_metadata)

    print(logprobs)
    # Merge the deferral probability back into the logprobs so that it can be returned
    # downstream.
    logprobs = _final_logits

    # Get the logprobs query results.
    prompt_logprobs, sample_logprobs = _get_logprobs(logprobs, input_metadata, sample_results)
    return _build_sampler_output(sample_results, input_metadata, prompt_logprobs, sample_logprobs)


Sampler.forward = forward

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()
engine = None


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


def request_output_to_dict(request_output):
    def output_to_dict(output) -> dict:
        return {
            "index": output.index,
            "text": output.text,
            "token_ids": output.token_ids,
            "cumulative_logprob": output.cumulative_logprob,
            "logprobs": output.logprobs if output.logprobs else None,
            "finish_reason": output.finish_reason,
        }

    return {
        "request_id": request_output.request_id,
        "prompt": request_output.prompt,
        "prompt_token_ids": request_output.prompt_token_ids,
        "prompt_logprobs": request_output.prompt_logprobs,  # Assuming PromptLogprobs has a to_dict method
        "outputs": [output_to_dict(output) for output in request_output.outputs],
        "finished": request_output.finished,
    }


class CaptureResetDeferralProb:
    def __init__(self):
        self._logits_cache = []

    def reset(self):
        self._logits_cache = []

    def cache(self, logits):
        self._logits_cache.append(logits.detach().cpu())

    def pop(self):
        return self._logits_cache.pop()

    def dump(self):
        print(self._logits_cache)
        output = [ele.tolist() for ele in self._logits_cache]
        self.reset()
        return output

    def __call__(self, token_ids, logits):
        print("caching logits")
        self.cache(logits)
        return logits


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    # prompt = request_dict.pop("prompt")
    prompt_token_ids = request_dict.pop("prompt_token_ids")
    stream = False
    logits_cache = CaptureResetDeferralProb()
    sampling_params = SamplingParams(
        **request_dict,
        # logits_processors=[logits_cache],
    )
    request_id = random_uuid()

    results_generator = engine.generate(
        prompt=None,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        request_id=request_id,
    )

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            text_outputs = [prompt + output.text for output in request_output.outputs]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    # return JSONResponse({**request_output_to_dict(final_output), **{"logits": logits_cache.dump()}})
    return JSONResponse(request_output_to_dict(final_output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    uvicorn.run(app, host=args.host, port=args.port, log_level="debug", timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
