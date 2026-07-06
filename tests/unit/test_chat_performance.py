import time
import sys
import os
from unittest.mock import MagicMock, patch

# Add project root and src to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
src_path = os.path.join(project_root, "src")
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# No global streamlit mock here to avoid polluting other tests


def test_rendering_complexity():
    """
    Verify that the rendering time of render_message increases linearly with
    the number of messages.
    """
    # Local mock setup to avoid global pollution of sys.modules
    mock_st = MagicMock()
    with patch.dict(sys.modules, {"streamlit": mock_st, "streamlit.components.v1": MagicMock()}):
        from ui.components.chat import render_message
        
        message_counts = [1, 10, 50, 100]
        results = {}

    def _slow_call(*args, **kwargs):
        time.sleep(0.001)
        return MagicMock()

    # We patch 'st' inside the 'ui.components.chat' module
    with patch('ui.components.chat.st') as mocked_st:
        # Assign slow calls to the mocked streamlit functions
        mocked_st.chat_message.side_effect = _slow_call
        mocked_st.container.side_effect = _slow_call
        mocked_st.markdown.side_effect = _slow_call
        mocked_st.popover.side_effect = _slow_call
        mocked_st.divider.side_effect = _slow_call
        mocked_st.caption.side_effect = _slow_call
        mocked_st.columns.side_effect = _slow_call
        mocked_st.button.side_effect = _slow_call

        print("\n[Performance Simulation: Rendering Complexity]")
        for count in message_counts:
            messages = [
                {"role": "user", "content": "Hello world, this is a test message."}
                for _ in range(count)
            ]

            start_time = time.perf_counter()
            for i, msg in enumerate(messages):
                render_message(
                    role=msg["role"],
                    content=msg["content"],
                    msg_index=i,
                    wrap_in_container=False,
                )
            end_time = time.perf_counter()

            results[count] = end_time - start_time
            print(f"Messages: {count:3} | Render Time: {results[count]:.6f}s")

    if len(results) >= 2:
        ratio = results[100] / results[1]
        print(f"\nScale factor (1 vs 100): {ratio:.2f}x")
        assert ratio > 10, "Rendering time should increase with the number of messages"

if __name__ == "__main__":
    test_rendering_complexity()
