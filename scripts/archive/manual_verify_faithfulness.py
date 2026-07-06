import re
import json

def extract_full_row(report_path, target_query_part):
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 테이블 행 추출 (멀티라인 지원)
    # 마크다운 테이블 행은 | 로 시작하고 끝남. 셀 내부에 줄바꿈이 있을 수 있으므로 주의.
    # 여기서는 단순화를 위해 행 단위 매칭 후 결합 시도
    lines = content.split('\n')
    table_lines = [l for l in lines if l.strip().startswith('|')]
    
    target_row = ""
    found = False
    for line in table_lines:
        if target_query_part in line:
            target_row = line
            found = True
            break
            
    if not found:
        return None, None, None

    cols = target_row.split('|')
    # | (empty) | user_input | retrieved_contexts | response | faithfulness | ...
    if len(cols) < 5:
        return None, None, None
        
    query = cols[1].strip()
    context = cols[2].strip()
    response = cols[3].strip()
    
    return query, context, response

if __name__ == "__main__":
    report = "reports/e2e_eval_report_20260218_144346.md"
    query_part = "CM3 모델이 이미지를 학습할 때"
    
    q, c, r = extract_full_row(report, query_part)
    
    if q:
        print(f"### [질문]\n{q}\n")
        print(f"### [컨텍스트 (원본)]\n{c[:1000]}...\n") # 너무 길면 생략
        print(f"### [답변]\n{r}\n")
        
        # 파일로 저장해서 자세히 볼 수 있게 함
        with open("manual_verification_data.txt", "w", encoding="utf-8") as f:
            f.write(f"QUERY: {q}\n\n")
            f.write(f"CONTEXT:\n{c}\n\n")
            f.write(f"RESPONSE:\n{r}\n")
    else:
        print("대상 행을 찾지 못했습니다.")
