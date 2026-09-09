import re
from pathlib import Path

def analyze_e2e_log(log_path: str):
    print(f"[*] Analyzing Log: {log_path}")
    if not Path(log_path).exists():
        print("[!] Log file not found.")
        return

    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    metrics = {
        "indexing_time": 0.0,
        "router_time": 0.0,
        "ttft": 0.0,
        "generation_speed": 0.0,
        "intent": "Unknown"
    }

    for line in lines:
        if "RAG 파이프라인 구축 완료" in line:
            match = re.search(r"소요: ([\d.]+)s", line)
            if match: metrics["indexing_time"] = float(match.group(1))
        
        if "[CHAT] [ROUTER] 의도 분석 완료" in line:
            match = re.search(r"\| (\w+) \| ([\d.]+)ms", line)
            if match:
                metrics["intent"] = match.group(1)
                metrics["router_time"] = float(match.group(2)) / 1000.0
        
        if "[LLM] 완료" in line:
            ttft_match = re.search(r"TTFT: ([\d.]+)s", line)
            speed_match = re.search(r"속도: ([\d.]+) tok/s", line)
            if ttft_match: metrics["ttft"] = float(ttft_match.group(1))
            if speed_match: metrics["generation_speed"] = float(speed_match.group(1))

    print("")
    print("========================================")
    print("       RAG PERFORMANCE REPORT")
    print("========================================")
    print(f"Target Intent:    {metrics['intent']}")
    print(f"Total Indexing:   {metrics['indexing_time']:.2f}s")
    print(f"Router Latency:   {metrics['router_time']:.2f}s")
    print(f"TTFT (First Tok): {metrics['ttft']:.2f}s")
    print(f"Gen Speed:        {metrics['generation_speed']:.1f} tok/s")
    print("========================================")

    if metrics['indexing_time'] > 30:
        print("[!] Warning: Indexing is slow.")
    if metrics['router_time'] > 5:
        print("[!] Warning: Router classification is slow.")
    if metrics['generation_speed'] < 10:
        print("[!] Warning: LLM generation is slow.")

if __name__ == "__main__":
    analyze_e2e_log("logs/test_e2e.log")