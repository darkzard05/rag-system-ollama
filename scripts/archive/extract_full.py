import re

def extract_full_row_robust(report_path, target_query_part):
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 마크다운 표의 행은 \n| 로 시작할 가능성이 높음
    # 하지만 셀 내부에 \n이 있으면 복잡해짐. 
    # 다행히 이 보고서는 한 행이 한 줄로 되어 있는 것으로 보임 (다만 줄이 매우 김)
    
    rows = content.split('\n')
    for row in rows:
        if target_query_part in row and row.strip().startswith('|'):
            cols = row.split('|')
            if len(cols) > 4:
                return cols[1].strip(), cols[2].strip(), cols[3].strip()
    return None, None, None

q, c, r = extract_full_row_robust("reports/e2e_eval_report_20260218_144346.md", "CM3 모델이 이미지를 학습할 때")

if q:
    # 줄바꿈이 \n 문자열로 들어있을 수 있으므로 변환
    r_clean = r.replace('\\n', '\n')
    c_clean = c.replace('\\n', '\n')
    
    print(f"--- QUERY ---\n{q}\n")
    print(f"--- RESPONSE (CLEANED) ---\n{r_clean[:1500]}...\n") # 출력 창 제한 고려
    
    with open("full_response_debug.txt", "w", encoding="utf-8") as f:
        f.write(r_clean)
    with open("full_context_debug.txt", "w", encoding="utf-8") as f:
        f.write(c_clean)
else:
    print("Row not found")
