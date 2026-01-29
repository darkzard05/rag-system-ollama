"""
Streaming Optimization Benchmark Simulation
Compares per-token streaming vs. batched streaming performance under simulated UI load.
"""

import asyncio
import time

# --- Simulation Parameters ---
TOKEN_COUNT = 1000
TOKEN_GENERATION_DELAY = (
    0.002  # LLM speed: 500 tokens/sec (simulated fast local/slow gpu)
)
UI_UPDATE_COST = 0.005  # Cost to update UI/send SSE per event


async def mock_event_handler(event_data):
    """Simulates the cost of processing an event (e.g., Streamlit rerun, network I/O)"""
    await asyncio.sleep(UI_UPDATE_COST)


async def simulate_legacy_streaming():
    """Simulates old behavior: Emit event for EVERY token"""
    start_time = time.perf_counter()
    event_count = 0

    for _ in range(TOKEN_COUNT):
        # Simulate LLM generating a token
        await asyncio.sleep(TOKEN_GENERATION_DELAY)
        token = "t"

        # Emit event immediately
        await mock_event_handler({"chunk": token})
        event_count += 1

    duration = time.perf_counter() - start_time
    return duration, event_count


async def simulate_optimized_streaming():
    """Simulates new behavior: Batch events (0.1s delay or 20 chars)"""
    start_time = time.perf_counter()
    event_count = 0
    buffer = ""
    last_emit_time = time.perf_counter()

    for _ in range(TOKEN_COUNT):
        # Simulate LLM generating a token
        await asyncio.sleep(TOKEN_GENERATION_DELAY)
        token = "t"
        buffer += token

        current_time = time.perf_counter()
        if (current_time - last_emit_time > 0.1) or len(buffer) >= 20:
            await mock_event_handler({"chunk": buffer})
            event_count += 1
            buffer = ""
            last_emit_time = current_time

    if buffer:
        await mock_event_handler({"chunk": buffer})
        event_count += 1

    duration = time.perf_counter() - start_time
    return duration, event_count


async def run_benchmark():
    print("🚀 Streaming Optimization Benchmark")
    print(
        f"Parameters: {TOKEN_COUNT} tokens, LLM Delay={TOKEN_GENERATION_DELAY}s, UI Cost={UI_UPDATE_COST}s"
    )
    print("-" * 60)

    # Run Legacy
    print("Running Legacy Simulation (Per-token)...")
    legacy_time, legacy_events = await simulate_legacy_streaming()
    print(f"Legacy: {legacy_time:.4f}s | Events: {legacy_events}")

    # Run Optimized
    print("\nRunning Optimized Simulation (Batched)...")
    opt_time, opt_events = await simulate_optimized_streaming()
    print(f"Optimized: {opt_time:.4f}s | Events: {opt_events}")

    # Result
    print("-" * 60)
    speedup = (legacy_time - opt_time) / legacy_time * 100
    overhead_reduction = (legacy_events - opt_events) / legacy_events * 100

    print(f"⚡ Speedup: {speedup:.2f}% faster")
    print(f"📉 Overhead Reduction: {overhead_reduction:.2f}% fewer events")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
