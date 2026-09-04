"""JSON 복구 헬퍼 (GENERATE 단계 파싱 실패 폴백).

모델이 값 내부에 이스케이프 없는 큰따옴표를 넣어 json.loads 가 실패하는
경우(Expecting ',' delimiter) 사용한다. 모든 추출은 escape 인식 스캐너로
수행해 D1(메타데이터 손실)/D2(text_span 절단)/D3(final_answer 절단) 를 방지.
"""

import json
import re
from typing import Any


def _extract_json_string(blob: str, start: int) -> tuple[str | None, int]:
    """시작 위치 ``start``(큰따옴표 직후)부터 닫는 큰따옴표까지 값을 읽는다.

    이스케이프된 ``\\"`` 는 건너뛴다. 스캔 종료 시 다음 읽기 시작 위치(닫는
    따옴표 바로 뒤)를 반환한다. 값을 찾지 못하면 (None, start) 를 반환한다.
    """
    i = start
    n = len(blob)
    # 키:"  값" 형태에서 콜론 뒤 공백/개행을 건너뛰고 값의 시작 큰따옴표로 이동
    while i < n and blob[i] in (" ", "\n", "\t", "\r"):
        i += 1
    if i >= n or blob[i] != '"':
        return None, start
    i += 1
    chars: list[str] = []
    while i < n:
        ch = blob[i]
        if ch == "\\":
            if i + 1 < n:
                nxt = blob[i + 1]
                if nxt == '"':
                    chars.append('"')
                    i += 2
                    continue
                if nxt == "\\":
                    chars.append("\\")
                    i += 2
                    continue
                chars.append(ch)
                i += 1
                continue
            chars.append(ch)
            i += 1
            continue
        if ch == '"':
            # 값 내부 큰따옴표(다음 문자가 공백/문자)와 닫는 따옴표(다음이
            # 구분자)를 구분: 구분자 직전만 값 종료로 본다.
            nxt = blob[i + 1] if i + 1 < n else ""
            if nxt in (":", ",", "}", "]"):
                return "".join(chars), i + 1
            chars.append('"')
            i += 1
            continue
        chars.append(ch)
        i += 1
    return None, start


def _recover_citations(blob: str) -> list[dict[str, Any]]:
    """깨진 JSON 에서 citations[] 를 객체 단위로 추출한다.

    각 citation 객체를 독립적으로 파싱하므로 doc_id/text_span 간 교차 매칭(D4)
    을 방지한다. section/page/score 도 실제 값을 보존한다(D1 해소).
    """
    citations: list[dict[str, Any]] = []
    obj_start = blob.find('"doc_id"')
    while obj_start != -1:
        obj_end = blob.find("}", obj_start)
        if obj_end == -1:
            break
        obj = blob[obj_start : obj_end + 1]
        # doc_id 는 실제로 fast_hash() 해시 문자열일 수 있으므로 따옴표 없는
        # 해시/숫자-문자 혼합 토큰도 포착한다 (스키마 int 와 실제 데이터 불일치 보정).
        doc_id_m = re.search(r'"doc_id"\s*:\s*("?)([0-9a-fA-F]+|\d+)("?)', obj)
        if not doc_id_m:
            obj_start = blob.find('"doc_id"', obj_end)
            continue
        doc_id = doc_id_m.group(2)
        span_m = re.search(r'"text_span"\s*:', obj)
        text_span: str = ""
        if span_m:
            _span, _ = _extract_json_string(obj, span_m.end())
            text_span = _span or ""
        section_m = re.search(r'"section"\s*:', obj)
        section: str = ""
        if section_m:
            _sec, _ = _extract_json_string(obj, section_m.end())
            section = _sec or ""
        page_m = re.search(r'"page"\s*:\s*(\d+)', obj)
        page = int(page_m.group(1)) if page_m else 0
        score_m = re.search(r'"score"\s*:\s*([\d.]+)', obj)
        score = float(score_m.group(1)) if score_m else 0.0
        citations.append(
            {
                "doc_id": doc_id,
                "text_span": text_span,
                "section": section,
                "page": page,
                "score": score,
            }
        )
        obj_start = blob.find('"doc_id"', obj_end)
    return citations


def _extract_partial_answer(blob: str) -> str | None:
    """깨진 JSON 에서 final_answer 값을 escape 인식 스캐너로 추출한다 (D3 해소)."""
    key_m = re.search(r'"final_answer"\s*:', blob)
    if not key_m:
        return None
    val_m = re.search(r'"(.*)', blob[key_m.end() :], re.DOTALL)
    if not val_m:
        return None
    # val_m.start() 는 blob[key_m.end():] 에서 첫 큰따옴표의 위치(0-based).
    # +1 하면 큰따옴표 바로 다음 문자가 되어 _extract_json_string 이 값을
    # 건너뛰므로, 값 시작 큰따옴표 위치 그대로 전달한다.
    val_start = key_m.end() + val_m.start()
    value, _ = _extract_json_string(blob, val_start)
    return value


def _repair_json(blob: str) -> str | None:
    """깨진 JSON 에서 값 내부 이스케이프 없는 큰따옴표를 보정한다.

    모델이 문자열 값 안에 raw 큰따옴표를 넣어 json.loads 가 실패하는 경우,
    값의 끝은 "닫는 큰따옴표 바로 다음 문자가 구분자(``,`` ``}`` ``]``
    공백/개행)인 지점"으로 추정해 내부 큰따옴표를 ``\\"`` 로 이스케이프한다.
    복구 후 json.loads 가 성공하면 보정본을, 실패하면 None 을 반환(그때만
    regex 폴백으로 진행). 의존성 없음.
    """
    i = 0
    n = len(blob)
    out: list[str] = []
    while i < n:
        ch = blob[i]
        if ch == "{":
            out.append(ch)
            i += 1
            continue
        if ch == "}":
            out.append(ch)
            i += 1
            continue
        # "doc_id": 뒤에 따옴표 없는 해시/식별자 토큰이 오는 경우 보정.
        # 실제 doc_id 는 fast_hash() 해시 문자열인데, 모델이 스키마(integer)를
        # 따르려다 따옴표 없이 넣으면 json.loads 가 실패한다. 토큰을 "..." 로 감싼다.
        if ch == '"' and blob[i : i + 8] == '"doc_id"':
            out.append('"doc_id"')
            i += 8
            # 콜론 + 공백 건너뛰기(콜론은 아래서 다시 붙인다)
            while i < n and blob[i] in (":", " ", "\n", "\t", "\r"):
                i += 1
            if i < n and blob[i] != '"':
                # 따옴표 없는 토큰(해시/숫자-문자 혼합) -> "key": "tok" 형태로 보정
                tok_start = i
                while i < n and blob[i] not in (",", "}", "]", "\n", " ", "\t", "\r"):
                    i += 1
                tok = blob[tok_start:i]
                out.append(':"' + tok + '"')
            else:
                # 이미 따옴표 있는 정상 형태면 ":" 만 붙이고 값은 기존 스캐너가 처리
                out.append(":")
                if i < n and blob[i] == '"':
                    out.append('"')
            continue
        if ch == '"':
            # 문자열 값 시작
            out.append(ch)
            i += 1
            while i < n:
                c = blob[i]
                if c == "\\":
                    out.append(c)
                    if i + 1 < n:
                        out.append(blob[i + 1])
                        i += 2
                    else:
                        i += 1
                    continue
                if c == '"':
                    # 닫는 따옴표 추정: 키명 뒤(":" 직전) 또는 값 끝("," / "}" / "]" 직전).
                    # 그 외(공백/문자 뒤)는 값 내부 큰따옴표로 보고 이스케이프한다.
                    nxt = blob[i + 1] if i + 1 < n else ""
                    if nxt in (":", ",", "}", "]"):
                        out.append(c)
                        i += 1
                        break
                    out.append('\\"')
                    i += 1
                    continue
                # 값 내부 raw 제어문자 이스케이프 (JSON 문자열 값은 \n/\r/\t 로 표기해야 함).
                # 모델이 실제 개행/탭을 그대로 넣으면 json.loads 가 실패하므로 보정한다.
                if c == "\n":
                    out.append("\\n")
                    i += 1
                    continue
                if c == "\r":
                    out.append("\\r")
                    i += 1
                    continue
                if c == "\t":
                    out.append("\\t")
                    i += 1
                    continue
                out.append(c)
                i += 1
            continue
        out.append(ch)
        i += 1
    repaired = "".join(out)
    try:
        json.loads(repaired)
        return repaired
    except (json.JSONDecodeError, ValueError):
        return None


def _strip_json_fence(raw: str) -> str:
    """LLM 응답에서 ```json / ``` 코드 펜스를 제거합니다. (R: Group 8)

    구조화(1.2)와 verify 노드에서 동일 스트리핑을 중복하던 것을 단일 헬퍼로 통합.
    """
    s = raw.strip()
    if s.startswith("```json"):
        s = s[7:]
    if s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()
