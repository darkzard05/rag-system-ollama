from unittest.mock import patch

from ui.components.chat import _infer_proc_phase


def test_infer_proc_phase_zero_translation_calls():
    """_infer_proc_phase가 t() 번역 함수나 get_phase_labels를 호출하지 않고 위상을 판정하는지 검증."""
    with (
        patch("ui.components.chat.get_phase_labels") as mock_get_labels,
        patch("ui.strings.t") as mock_t,
    ):
        # 1. 문서 검색 위상 판정
        assert _infer_proc_phase("관련 문서 검색 중...") == 0
        # 2. 증거 수집 위상 판정
        assert _infer_proc_phase("문맥 context 리랭크 수행 중") == 1
        # 3. 답변 생성 위상 판정
        assert _infer_proc_phase("최종 답변 생성 시작") == 2
        # 4. 미매칭 텍스트
        assert _infer_proc_phase("알 수 없는 시스템 작업") == -1

        # 번역 함수 호출이 일체 발생하지 않아야 함
        mock_get_labels.assert_not_called()
        mock_t.assert_not_called()
