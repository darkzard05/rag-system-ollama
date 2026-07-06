## Development Workflow

Rules:
- 모든 코드 작성 및 수정은 사용자의 명시적인 허가(Directive)가 있기 전까지 엄격히 금지됩니다. 허가 전에는 분석 및 제안만 수행하십시오.
- **코드 수정 후 반드시 검증:** 모든 파일 수정 후에는 반드시 `ruff check <수정된_파일_경로>`를 실행하여 문법 오류 및 정의되지 않은 이름(Undefined names)을 확인하고 그 결과를 보고해야 합니다.

## Context7 Usage

Rules:
- 라이브러리, 프레임워크, API 참조가 필요한 경우 반드시 `Context7 (Find-docs)` 스킬을 사용하여 최신 정보를 확인해야 합니다.
- 학습 데이터에 의존하지 말고, 최신 API 명세나 설정 옵션을 항상 검증하십시오.

## Git & Commits

Rules:
- 커밋 메시지는 항상 영어로 작성해야 합니다. (Commit messages must always be in English.)

## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- For cross-module "how does X relate to Y" questions, prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"` over grep — these traverse the graph's EXTRACTED + INFERRED edges instead of scanning files
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost)
