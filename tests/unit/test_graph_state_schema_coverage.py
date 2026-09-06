"""구조적 스키마 커버리지 가드 테스트 (FIX-7).

LangGraph 1.6.0 은 노드가 반환한 **미선언 state 키를 조용히 드롭**한다.
이 결함(short_query/cached_response)을 영구히 막기 위해, 모든 그래프 노드의
리턴 dict 키가 `GraphState` 스키마 선언 키 집합의 부분집합인지 AST 파싱으로 검증한다.

노드 반환은 반드시 **리터럴 dict**로 작성해야 한다 (동적 dict 리턴:
``d = {}; d["k"] = ...; return d``은 리터럴 키 수집을 우회함).
현재 5개 노드 모두 리터럴 리턴을 사용한다.
"""

import ast
import sys
from pathlib import Path

# --- 경로 설정 ---
BASE_DIR = Path(__file__).parent.parent.parent.absolute()
SRC_DIR = BASE_DIR / "src"
for p in [str(BASE_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest

# build_graph() 에 등록된 노드 함수명
# (graph_builder.py:1681-1685 add_node 첫 인자와 일치해야 함)
GRAPH_NODE_FUNCS = {
    "preprocess",
    "retrieve_and_rerank",
    "grade_documents",
    "rewrite_query",
    "generate",
}

GRAPH_BUILDER_PATH = SRC_DIR / "core" / "graph" / "graph_builder.py"
SCHEMAS_PATH = SRC_DIR / "api" / "schemas.py"

# Step 2: preprocess 는 core.graph._preprocess 로 이동됨.
# Step 3: retrieve_and_rerank 는 core.graph._retrieve 로 이동됨.
# Step 4: grade_documents / rewrite_query 는 core.graph._grade 로 이동됨.
# Step 5: generate / format_context 등은 core.graph._generate 로 이동됨.
# 노드별 소스 후보 파일 (추후 분리되는 모듈은 여기에 추가).
NODE_FUNC_SOURCE_PATHS: dict[str, list[Path]] = {
    "preprocess": [
        SRC_DIR / "core" / "graph" / "_preprocess.py",
        GRAPH_BUILDER_PATH,
    ],
    "retrieve_and_rerank": [
        SRC_DIR / "core" / "graph" / "_retrieve.py",
        GRAPH_BUILDER_PATH,
    ],
    "grade_documents": [
        SRC_DIR / "core" / "graph" / "_grade.py",
        GRAPH_BUILDER_PATH,
    ],
    "rewrite_query": [
        SRC_DIR / "core" / "graph" / "_grade.py",
        GRAPH_BUILDER_PATH,
    ],
    "generate": [
        SRC_DIR / "core" / "graph" / "_generate.py",
        GRAPH_BUILDER_PATH,
    ],
}


def _collect_return_keys(
    func_def: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[str] | None:
    """함수 본문의 모든 `return <dict>` 리터럴 키 집합을 수집한다.

    - 리터럴 dict 가 아닌 `return`(동적 dict, 변수/리터럴이 아닌 값)은
      키 수집을 우회할 수 있으므로 `None` 을 반환한다.
    - 리터럴 dict 는 비어 있어도(`return {}`) 정상으로 취급한다.
    """
    keys: set[str] = set()
    has_literal_dict_return = False
    has_non_literal_return = False

    class RetVisitor(ast.NodeVisitor):
        def visit_Return(self, node: ast.Return) -> None:
            nonlocal has_literal_dict_return, has_non_literal_return
            if node.value is None:
                return
            if isinstance(node.value, ast.Dict):
                has_literal_dict_return = True
                for k in node.value.keys:
                    if isinstance(k, ast.Constant):
                        keys.add(str(k.value))
            else:
                has_non_literal_return = True

    RetVisitor().visit(func_def)
    if has_non_literal_return and not has_literal_dict_return:
        return None
    return keys


def _graph_state_declared_keys() -> set[str]:
    """GraphState TypedDict 의 선언 키 집합 (pathlib + AST)."""
    src = SCHEMAS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "GraphState":
            keys = set()
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(
                    stmt.target, ast.Name
                ):
                    keys.add(stmt.target.id)
            return keys
    raise AssertionError("GraphState TypedDict 를 찾을 수 없습니다.")


def test_graph_state_declares_cached_response_and_short_query():
    """결함의 핵심: FIX-1 전에는 cached_response/short_query 미선언 → red.

    오류 요지: preprocess 가 이 키들을 반환하지만 스키마에 없어 LangGraph 가
    조용히 드롭한다. 이 테스트는 스키마에 두 키가 선언됐는지 직접 검증한다.
    """
    declared = _graph_state_declared_keys()
    missing = {"cached_response", "short_query"} - declared
    assert not missing, (
        f"GraphState 에 미선언 키: {sorted(missing)}. "
        f"LangGraph 가 노드 반환 키를 조용히 드롭해 "
        f"short_query fast path 와 cached_response 캐시 경로가 동작하지 않습니다."
    )


@pytest.mark.parametrize(
    "func_name", sorted(GRAPH_NODE_FUNCS), ids=lambda n: f"node={n}"
)
def test_every_node_return_keys_are_declared_in_schema(func_name: str):
    """그래프 노드의 모든 반환 dict 키 ⊆ GraphState 스키마 집합 (부분집합)."""
    func_def = None
    for src_path in NODE_FUNC_SOURCE_PATHS.get(func_name, [GRAPH_BUILDER_PATH]):
        tree = ast.parse(src_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == func_name
            ):
                func_def = node
                break
        if func_def is not None:
            break
    if func_def is None:
        pytest.fail(f"스플릿 후 노드 함수 {func_name!r} 을 소스에서 찾지 못했습니다.")

    decl_keys = _graph_state_declared_keys()
    returned_keys = _collect_return_keys(func_def)

    assert returned_keys is not None, (
        f"{func_name} 에 리터럴 dict 반환이 없습니다 (동적 dict 우회 위험)."
    )
    undeclared = returned_keys - decl_keys
    assert not undeclared, (
        f"노드 {func_name} 이 미선언 키를 반환: {sorted(undeclared)}. "
        f"LangGraph 채널 계층이 이를 조용히 드롭합니다."
    )


def test_graph_state_declaration_captured():
    """GraphState 선언 키가 실제 총량과 일치하는지 스냅샷 (성능 회귀 감지용)."""
    declared = _graph_state_declared_keys()
    # 스키마가 커지면 이 테스트를 함께 갱신한다 (의도적 변경 추적).
    expected = {
        "input",
        "intent",
        "route",
        "search_queries",
        "relevant_docs",
        "response",
        "thought",
        "performance",
        "search_weights",
        "is_cached",
        "cached_response",
        "short_query",
        "retry_count",
    }
    assert declared == expected, f"GraphState 키 집합 불일치: {sorted(declared)}"
