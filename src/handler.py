import os
import logging
import multiprocessing as mp
import runpod
import json
from torch.cuda import device_count
from utils import JobInput
from engine import vLLMEngine, OpenAIvLLMEngine
from engine_args import get_engine_args

# Read models and GPU assignments from env
model_list = [m.strip() for m in os.getenv("MODEL_IDS", "").split(";") if m.strip()]
# gpu_assignments = [g.strip() for g in os.getenv("MODEL_GPUS", "").split(";") if g.strip()]

def engine_infer(model_id, device_ids, job):
    # Set per-process CUDA environment
    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids
    from engine import vLLMEngine, OpenAIvLLMEngine  # re-import in subprocess
    from engine_args import get_engine_args

    engine_args = get_engine_args(
        model=model_id,
        dtype="float16",
        gpu_id=device_ids,
        tensor_parallel_size=(len(device_ids.split(",")) if device_ids and "," in device_ids else 1)
    )
    vllm_engine = vLLMEngine(engine_args=engine_args)
    openai_engine = OpenAIvLLMEngine(vllm_engine)
    result = []
    for batch in openai_engine.generate(job):
        result.append(batch)
    return result

# ---------- multiprocessing wrapper ----------
def process_all_jobs(jobs_dict):
    """
    jobs_dict = {
        "input1": {...},
        "input2": {...},
        ...
    }
    """
    # Defensive: JSON string → dict
    if isinstance(jobs_dict, str):
        jobs_dict = json.loads(jobs_dict)

    mp_ctx = mp.get_context("spawn")
    pool   = mp_ctx.Pool(processes=len(jobs_dict))
    async_results = []

    total_gpu = device_count()

    for name, job in jobs_dict.items():
        # Defensive: job itself might be a JSON string
        if isinstance(job, str):
            job = json.loads(job)

        try:
            model_id = job["model_name"]
            gpu_ids  = str(job["gpu_ids"])
        except (KeyError, TypeError) as e:
            logging.error(f"Malformed job payload for '{name}': {e}")
            async_results.append(None)
            continue

        # Fallback in case someone passes an out-of-range GPU id
        if not gpu_ids or int(gpu_ids.split(",")[0]) >= total_gpu:
            logging.error(f"Job '{name}' requested invalid gpu_ids='{gpu_ids}'")
            async_results.append(None)
            continue

        async_results.append(
            pool.apply_async(engine_infer, args=(model_id, gpu_ids, job))
        )

    pool.close(); pool.join()

    results = []
    for a in async_results:
        if a is None:
            results.append(None)
            continue
        try:
            results.append(a.get(timeout=300))
        except Exception as e:
            logging.error(f"Inference failed: {e}")
            results.append(None)
    return results


async def handler(job):
    jobs = job["input"]  # jobs is now a dict of job_name: job_detail
    results = process_all_jobs(jobs)
    for result in results:
        if result is not None:
            for batch in result:
                yield batch

runpod.serverless.start({
    "handler": handler,
    "concurrency_modifier": lambda x: len(model_list),
    "return_aggregate_stream": True,
})
