# tests/verify_context.py
import asyncio
from core.session.context import ContextManager


async def worker(name: str, session_id: str):
    print(f"Worker {name}: Setting session_id to {session_id}")
    ContextManager.set_current_session_id(session_id)

    # Simulate some async work
    await asyncio.sleep(0.1)

    current_id = ContextManager.get_current_session_id()
    print(f"Worker {name}: Retrieved session_id {current_id}")

    if current_id != session_id:
        raise AssertionError(f"Worker {name}: Expected {session_id}, got {current_id}")


async def main():
    print("Starting ContextVar verification...")

    # Test 1: Isolation across async tasks
    tasks = [
        worker("A", "session-aaa"),
        worker("B", "session-bbb"),
        worker("C", "session-ccc"),
    ]

    await asyncio.gather(*tasks)
    print("✅ Async isolation test passed!")

    # Test 2: Default value
    # We need a fresh context or a way to clear it.
    # Since we are in the same thread, we can't easily 'clear' the main task's context
    # without setting it to None.
    ContextManager.set_current_session_id("main-session")
    print(f"Main: set to {ContextManager.get_current_session_id()}")

    # Run a task that doesn't set its own ID - it should inherit the parent's context
    async def inherit_worker():
        print(f"InheritWorker: Retrieved {ContextManager.get_current_session_id()}")
        if ContextManager.get_current_session_id() != "main-session":
            raise AssertionError("InheritWorker should have inherited main-session")

    await inherit_worker()
    print("✅ Context inheritance test passed!")


if __name__ == "__main__":
    asyncio.run(main())
