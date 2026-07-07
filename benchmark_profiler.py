import asyncio
import random
import time
import uuid
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

# 1. Define a realistic mix of prompts (short, medium, long)
PROMPTS = [
    "Hello, how are you?",
    "Explain the concept of entropy in physics.",
    "Write a Python script to scrape a website using BeautifulSoup.",
    "Summarize the plot of the Matrix.",
    "What is the capital of France?",
    "Write a long essay about the geopolitical impact of renewable energy in the 21st century.",
    "Translate the following sentence to Spanish: The quick brown fox jumps over the lazy dog.",
    "Debug this imaginary code: def foo(): return 1 / 0",
] * 4  # Multiplied to simulate a larger batch of 32 requests


def random_uuid() -> str:
    return str(uuid.uuid4())


async def process_request(engine, prompt, sampling_params):
    """Processes a single request and measures token timestamps."""
    request_id = random_uuid()
    start_time = time.perf_counter()
    first_token_time = None

    # Initialize the generator stream
    stream = engine.generate(prompt, sampling_params, request_id)

    async for request_output in stream:
        # Capture the exact moment the first token arrives
        if first_token_time is None and request_output.outputs[0].token_ids:
            first_token_time = time.perf_counter()

    end_time = time.perf_counter()
    output_tokens = len(request_output.outputs[0].token_ids)

    return {
        "prompt_length_chars": len(prompt),
        "output_tokens": output_tokens,
        "start_time": start_time,
        "first_token_time": first_token_time or end_time,
        "end_time": end_time,
    }


async def simulate_traffic(engine, prompts, target_qps):
    """Simulates real-world traffic using a Poisson distribution for arrival times."""
    tasks = []
    sampling_params = SamplingParams(temperature=0.7, max_tokens=256)

    print(f"Starting benchmark simulation at {target_qps} QPS...")

    for prompt in prompts:
        # Poisson arrival time (standard for load testing)
        delay = random.expovariate(target_qps)
        await asyncio.sleep(delay)

        # Fire off the request asynchronously
        task = asyncio.create_task(
            process_request(engine, prompt, sampling_params)
        )
        tasks.append(task)

    # Wait for all simulated user requests to finish
    results = await asyncio.gather(*tasks)
    return results


async def main():
    # --- Main Execution Block ---
    # Configure the Async Engine
    engine_args = AsyncEngineArgs(
        model="Qwen/Qwen2.5-1.5B-Instruct-AWQ",  # Swap to "Qwen/Qwen2-1.5B-Instruct" for baseline tests
        dtype="float16",
        quantization="awq",  # Remove or set to None when testing the unquantized baseline model
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
    )

    print("Loading Async vLLM Engine...")
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # Run the simulation (e.g., simulating 4 requests per second)
    target_qps = 4.0
    results = await simulate_traffic(engine, PROMPTS, target_qps)
    print("Simulation complete! Processing metrics...")

    # 2. Process the raw data into a DataFrame
    df = pd.DataFrame(results)

    # Calculate core LLM metrics
    df["ttft_ms"] = (df["first_token_time"] - df["start_time"]) * 1000
    df["tpot_ms"] = (
        (df["end_time"] - df["first_token_time"])
        / (df["output_tokens"] - 1).clip(lower=1)
    ) * 1000
    df["e2e_latency_s"] = df["end_time"] - df["start_time"]
    df["throughput_tok_s"] = df["output_tokens"] / df["e2e_latency_s"]

    # Print clean summary table to console
    print("\n--- Benchmarking Summary ---")
    print(f"Total Requests: {len(df)}")
    print(f"Median TTFT (Time-to-First-Token): {df['ttft_ms'].median():.2f} ms")
    print(
        f"Median TPOT (Time-Per-Output-Token): {df['tpot_ms'].median():.2f} ms"
    )
    print(f"Median E2E Latency: {df['e2e_latency_s'].median():.2f} s")
    print(
        f"Average Throughput: {df['throughput_tok_s'].mean():.2f} tokens/sec per request\n"
    )

    # Optional: Save raw results to disk
    df.to_csv("llm_benchmark_results.csv", index=False)

    # 3. Build the Visual Dashboard
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: TTFT Distribution
    sns.histplot(df["ttft_ms"], bins=15, kde=True, ax=axes[0], color="coral")
    axes[0].set_title("Time-to-First-Token (TTFT) Distribution")
    axes[0].set_xlabel("TTFT (ms)")
    axes[0].set_ylabel("Frequency")

    # Plot 2: Inter-Token Latency (TPOT) Distribution
    sns.histplot(df["tpot_ms"], bins=15, kde=True, ax=axes[1], color="teal")
    axes[1].set_title("Time-Per-Output-Token (TPOT) Distribution")
    axes[1].set_xlabel("TPOT (ms)")
    axes[1].set_ylabel("Frequency")

    # Plot 3: E2E Latency vs Output Length
    sns.scatterplot(
        data=df,
        x="output_tokens",
        y="e2e_latency_s",
        ax=axes[2],
        color="indigo",
        s=100,
    )
    axes[2].set_title("End-to-End Latency vs Output Tokens")
    axes[2].set_xlabel("Output Tokens Generated")
    axes[2].set_ylabel("Latency (seconds)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # vLLM's AsyncEngine requires an active event loop running
    asyncio.run(main())
