"""
Streaming Optimization Benchmark (Focus on TTFT)
Compares TTFT (Time-to-First-Token) between legacy batched streaming and the new 'First Token Bypass' optimization.
Simulates a slow LLM (0.5 tokens/sec).
"""

import asyncio
import time

# --- Simulation Parameters ---
SLOW_TOKEN_DELAY = 2.0  # 0.5 tokens/sec (Very slow local CPU inference)
TOTAL_TOKENS = 5
UI_UPDATE_COST = 0.005


async def mock_event_handler(event_data):
    # Simulates UI update
    await asyncio.sleep(UI_UPDATE_COST)


async def simulate_legacy_batched():
    """Old: Batch even the first token (PURE char count buffer focus)"""
    start_time = time.perf_counter()
    first_token_time = None
    buffer = ""

    for i in range(20):  # 20 tokens to hit buffer limit
        await asyncio.sleep(0.5)  # 2 tokens/sec
        token = "T"
        buffer += token

        # Only char count condition
        if len(buffer) >= 20:
            await mock_event_handler({"chunk": buffer})
            if first_token_time is None:
                first_token_time = time.perf_counter() - start_time
            buffer = ""

    return first_token_time


async def simulate_bypass_logic():
    """New: Bypass for the first token, then use char buffer"""
    start_time = time.perf_counter()
    first_token_time = None
    buffer = ""
    is_first = True

    for i in range(20):
        await asyncio.sleep(0.5)
        token = "T"
        buffer += token

        if is_first or len(buffer) >= 20:
            await mock_event_handler({"chunk": buffer})
            if first_token_time is None:
                first_token_time = time.perf_counter() - start_time
            buffer = ""
            is_first = False

    return first_token_time


async def run_test():
    print("🚀 TTFT Potential Delay Benchmark (Char Buffer Focus)")
    print("-" * 60)

    t_legacy = await simulate_legacy_batched()
    print(f"Legacy (Wait for 20 chars): {t_legacy:.2f}s")

    t_opt = await simulate_bypass_logic()
    print(f"Optimized (Bypass 1st):     {t_opt:.2f}s")

    print("-" * 60)
    reduction = t_legacy - t_opt
    print(f"⚡ Perceived Speedup: {reduction:.2f} seconds EARLIER!")


if __name__ == "__main__":
    asyncio.run(run_test())
