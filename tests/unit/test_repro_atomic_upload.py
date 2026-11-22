"""Regression: atomic PDF persist must prevent torn reads.

``Path.write_bytes`` truncates in place, so a concurrent reader (build
pipeline) could observe a partially written file. ``persist_pdf_atomically``
(tmp + os.replace) must guarantee readers see only complete versions.
"""

import threading


def test_atomic_persist_reader_only_sees_complete_versions(tmp_path) -> None:
    """수정 후 회귀 테스트: persist_pdf_atomically 동시 쓰기 중 리더는 완전본만 본다."""
    from api.routes_docs import persist_pdf_atomically

    target = tmp_path / "abc.pdf"
    payload_a = b"%PDF-A" + b"A" * 1_000_000
    payload_b = b"%PDF-B" + b"B" * 1_000_000
    persist_pdf_atomically(target, payload_a)

    torn: list[bytes] = []
    stop = threading.Event()

    def writer() -> None:
        for _ in range(20):
            persist_pdf_atomically(target, payload_b)
            persist_pdf_atomically(target, payload_a)

    def reader() -> None:
        while not stop.is_set():
            try:
                data = target.read_bytes()
            except OSError:
                continue
            if data not in (payload_a, payload_b):
                torn.append(data[:20])
                return

    t_reader = threading.Thread(target=reader)
    t_writer = threading.Thread(target=writer)
    t_reader.start()
    t_writer.start()
    t_writer.join()
    stop.set()
    t_reader.join(timeout=10)
    assert not torn, f"원자적 영속화에도 찢긴 읽기 발생: {torn!r}"
    assert target.read_bytes() in (payload_a, payload_b)
    leftovers = list(tmp_path.glob("*.tmp"))
    assert not leftovers, f"임시파일 잔류: {leftovers!r}"
