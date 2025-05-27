import os
import runpod
from utils import JobInput
from engine import vLLMEngine, OpenAIvLLMEngine
from engine_args import get_engine_args

# vllm_engine = vLLMEngine()
# OpenAIvLLMEngine = OpenAIvLLMEngine(vllm_engine)
OpenAIvLLMEngines = {}

# Read model IDs and GPU assignments from env (or config)
model_list = os.getenv("MODEL_IDS", "").split(";")
gpu_assignments = os.getenv("MODEL_GPUS", "").split(";")
loaded_models = {}

for idx, model_id in enumerate(model_list):
    model_id = model_id.strip()
    if not model_id:
        continue
    # Determine GPU(s) for this model
    device_ids = gpu_assignments[idx].strip() if idx < len(gpu_assignments) else ""
    if device_ids:
        # Limit visible devices to the specified ones for this model
        os.environ["CUDA_VISIBLE_DEVICES"] = device_ids
    # Initialize the model using vLLM LLM class
    engine_args = get_engine_args(
        model=model_id,
        dtype="float16",               # load weights in FP16
        gpu_id=device_ids,
        # trust_remote_code=True,        # allow custom model code if needed
        tensor_parallel_size=(len(device_ids.split(",")) if device_ids and "," in device_ids else 1)
    )
    vllm_engine = vLLMEngine(engine_args=engine_args)
    OpenAIvLLMEngine_model = OpenAIvLLMEngine(vllm_engine)
    OpenAIvLLMEngines[model_id] = OpenAIvLLMEngine_model

async def process_jobs(jobs):
    for job in jobs:
        if job["model_name"] == model_list[0]:
            task = OpenAIvLLMEngines[model_list[0]].generate(job)
            async for batch in results_generator:
                yield batch
    return await asyncio.gather(*tasks)

async def handler(job):
    job_input = JobInput(job["input"])
    # engine = OpenAIvLLMEngine if job_input.openai_route else vllm_engine

    results = await process_jobs(job_input)
    for result in results:
        yield result
    # results_generator = engine.generate(job_input)
    async for batch in results_generator:
        yield batch

runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": lambda x: vllm_engine.max_concurrency,
        "return_aggregate_stream": True,
    }
)