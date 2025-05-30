import os
import logging
import multiprocessing as mp
import runpod
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

def process_all_jobs(jobs):
    mp_ctx = mp.get_context('spawn')
    pool = mp_ctx.Pool(processes=len(jobs))
    async_results = []
    num_gpu = device_count()

    for job_key, job in jobs.items():
        model_id = job["model_name"]
        gpu_ids = str(job["gpu_ids"])
        gpu_len = gpu_ids.count(",") + 1
        if model_id not in model_list:
            async_results.append(None)
            continue
        device_ids = gpu_ids if gpu_len < len(num_gpu) else ""
        async_result = pool.apply_async(engine_infer, args=(model_id, device_ids, job))
        async_results.append(async_result)

    pool.close()
    pool.join()

    results = []
    for async_result in async_results:
        if async_result is None:
            results.append(None)
            continue
        try:
            results.append(async_result.get(timeout=300))
        except Exception as e:
            logging.error(f"Inference failed: {e}")
            results.append(None)
    return results

async def handler(job):
    job_input = JobInput(job["input"])
    results = process_all_jobs(job_input)
    for result in results:
        if result is not None:
            for batch in result:
                yield batch

runpod.serverless.start({
    "handler": handler,
    "concurrency_modifier": lambda x: len(model_list),
    "return_aggregate_stream": True,
})
