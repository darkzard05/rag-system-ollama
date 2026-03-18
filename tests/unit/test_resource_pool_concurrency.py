import asyncio
import pytest
import threading
from src.core.resource_pool import ResourcePool

def run_in_new_loop(coro, result_container):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result_container["result"] = loop.run_until_complete(coro)
    except Exception as e:
        result_container["error"] = e
    finally:
        loop.close()

@pytest.mark.asyncio
async def test_resource_pool_event_loop_mismatch():
    """
    Demonstrates RuntimeError when ResourcePool (singleton) uses asyncio.Lock
    and is accessed from different event loops.
    """
    pool = ResourcePool()
    
    # Access pool in the current loop (this will initialize self._lock_obj)
    await pool.register("test1", "vs1", "bm1")
    
    # Define a task to run in a DIFFERENT event loop
    async def access_pool():
        # This should fail because pool._lock is bound to the previous loop
        await pool.get("test1")

    # Run in a separate thread with its own event loop
    result_container = {"result": None, "error": None}
    thread = threading.Thread(target=run_in_new_loop, args=(access_pool(), result_container))
    thread.start()
    thread.join()
    
    error = result_container["error"]
    assert error is None, f"Expected no error but got: {error}"

if __name__ == "__main__":
    # Manually run if needed
    try:
        asyncio.run(test_resource_pool_event_loop_mismatch())
    except Exception as e:
        print(f"Caught expected error: {e}")
