from html import escape

def test_xss_vulnerability_reproduction():
    """
    LLM 응답(thought, content 등)에 XSS 페이로드가 포함되었을 때 
    HTML 이스케이프 없이 직접 주입하면 보안 위험이 발생함을 검증합니다.
    """
    payload = "</div><script>alert('xss')</script>"
    
    # 현재 취약한 방식: f-string 직접 주입
    vulnerable_html = f'<div class="thought-container">{payload}</div>'
    
    # 취약성 확인: 스크립트 태그가 그대로 존재하고 div가 조기 종료됨
    assert "</div><script>" in vulnerable_html
    print("❌ 취약한 HTML 생성됨 (XSS 위험)")

    # 수정될 방식: html.escape 사용
    safe_payload = escape(payload)
    safe_html = f'<div class="thought-container">{safe_payload}</div>'
    
    # 안전함 확인: 스크립트 태그가 이스케이프됨
    assert "&lt;/div&gt;&lt;script&gt;" in safe_html
    assert "<script>" not in safe_html
    print("✅ 안전한 HTML 생성됨 (이스케이프 확인)")

if __name__ == "__main__":
    try:
        test_xss_vulnerability_reproduction()
        print("테스트 완료")
    except AssertionError as e:
        print(f"테스트 실패: {e}")
