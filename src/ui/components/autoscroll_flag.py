"""Flag-read autoscroll helper (pure Python, no Streamlit)."""


def flag_read(
    is_generating_answer: bool, active_stream_msg_id: str, msg_id: str
) -> bool:
    """Return True only while the given message is the active stream."""
    return (
        bool(is_generating_answer)
        and bool(active_stream_msg_id)
        and active_stream_msg_id == msg_id
    )
