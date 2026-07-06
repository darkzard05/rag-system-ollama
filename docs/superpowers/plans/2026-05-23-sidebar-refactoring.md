# Sidebar 및 UI 구조 개선 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 사이드바의 관심사 분리(SoC), CSS 외부 관리, VRAM 정리 로직 안정화 및 전반적인 UI 유지보수성 향상

**Architecture:** 
1. 모델 필터링 로직을 `ModelManager`로 이동하여 UI와 비즈니스 로직을 분리합니다.
2. `src/ui/assets/style.css`를 생성하여 인라인 CSS를 관리하고, `ui.py`에서 이를 로드합니다.
3. `sidebar.py`를 리팩토링하여 추출된 로직을 사용하고, VRAM 정리 로직의 실행 위치를 최적화합니다.

**Tech Stack:** Python, Streamlit, CSS

---

### Task 1: Core 모델 필터링 로직 추출

**Files:**
- Modify: `src/core/model_loader.py`
- Test: `tests/unit/core/test_model_loader_filtering.py`

- [ ] **Step 1: `ModelManager`에 모델 필터링 메서드 추가**

```python
    @classmethod
    def get_filtered_models(cls, available_models: list[str]) -> dict[str, list[str]]:
        """모델 목록을 LLM과 임베딩 모델로 분류하여 반환합니다."""
        from common.config import AVAILABLE_EMBEDDING_MODELS, DEFAULT_EMBEDDING_MODEL, DEFAULT_OLLAMA_MODEL
        
        safe_models = [m for m in available_models if m and "---" not in str(m)]
        embed_keywords = ["embed", "bge", "nomic", "mxbai", "snowflake"]

        embedding_candidates = [
            m for m in safe_models if any(kw in str(m).lower() for kw in embed_keywords)
        ]
        actual_embeddings = sorted(set(AVAILABLE_EMBEDDING_MODELS + embedding_candidates))
        if DEFAULT_EMBEDDING_MODEL not in actual_embeddings:
            actual_embeddings.append(DEFAULT_EMBEDDING_MODEL)
        actual_embeddings.sort()

        llm_candidates = [m for m in safe_models if m not in embedding_candidates]
        actual_llms = llm_candidates if llm_candidates else [DEFAULT_OLLAMA_MODEL]
        if DEFAULT_OLLAMA_MODEL not in actual_llms:
            actual_llms.append(DEFAULT_OLLAMA_MODEL)
        actual_llms.sort()

        return {
            "llm": actual_llms,
            "embedding": actual_embeddings
        }
```

- [ ] **Step 2: 필터링 로직 유닛 테스트 작성**
- [ ] **Step 3: 테스트 실행 및 검증**
- [ ] **Step 4: Commit**

### Task 2: UI Asset 관리 및 CSS 외부화

**Files:**
- Create: `src/ui/assets/style.css`
- Modify: `src/ui/ui.py`

- [ ] **Step 1: `src/ui/assets/style.css` 생성 및 CSS 이동**
  - `ui.py`의 기존 CSS와 `settings-label`, `settings-sublabel` 스타일을 통합합니다.

- [ ] **Step 2: `ui.py` 수정하여 외부 CSS 로드 로직 구현**

```python
def inject_custom_css():
    import os
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
```

- [ ] **Step 3: Commit**

### Task 3: Sidebar 리팩토링 및 VRAM 정리 로직 개선

**Files:**
- Modify: `src/ui/components/sidebar.py`

- [ ] **Step 1: 추출된 `get_filtered_models` 사용하도록 수정**
- [ ] **Step 2: VRAM 정리 로직을 별도 핸들러 함수로 분리**
- [ ] **Step 3: 버튼 클릭 핸들러 내부에 `clear_vram` 직접 호출**

```python
def handle_vram_clear():
    from common.utils import sync_run
    from core.model_loader import ModelManager
    sync_run(ModelManager.clear_vram())
    st.toast("VRAM 정리 완료")

# ... 사이드바 내부에서
if st.button("Clear VRAM", ...):
    handle_vram_clear()
```

- [ ] **Step 4: Commit**

### Task 4: 최종 통합 검증

- [ ] **Step 1: Streamlit 앱 실행 및 사이드바 기능 확인**
- [ ] **Step 2: 모델 선택 목록 정상 표시 여부 확인**
- [ ] **Step 3: VRAM 정리 및 Reset 기능 동작 확인**
- [ ] **Step 4: Commit 및 완료**
