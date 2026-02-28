import time
from api.streaming_handler import TokenStreamBuffer

def test_first_token_immediate_flush():
    buffer = TokenStreamBuffer(buffer_size=10)
    
    print("1. 첫 번째 토큰 추가 중...")
    result1 = buffer.add_token("Hello")
    if result1 == "Hello":
        print("✅ SUCCESS: 첫 번째 토큰이 즉시 플러시되었습니다.")
    else:
        print(f"❌ FAILURE: 첫 번째 토큰이 플러시되지 않았습니다. (결과: {result1})")

    print("\n2. 두 번째 토큰 추가 중...")
    result2 = buffer.add_token(" world")
    if result2 is None:
        print("✅ SUCCESS: 두 번째 토큰은 버퍼링되었습니다.")
    else:
        print(f"❌ FAILURE: 두 번째 토큰이 즉시 플러시되었습니다. (결과: {result2})")

    print("\n3. 8개의 토큰 추가 중 (총 10개 채우기)...")
    for i in range(7): # 1(Hello)+1(world)+7 = 9
        buffer.add_token(f" token_{i}")
    
    print("\n10번째 토큰 추가 중...")
    result10 = buffer.add_token(" final")
    if result10 and "final" in result10:
        print(f"✅ SUCCESS: 10개 토큰이 모여 플러시되었습니다. (결과: {result10})")
    else:
        print(f"❌ FAILURE: 10개 토큰이 모였음에도 플러시되지 않았습니다. (결과: {result10})")

if __name__ == "__main__":
    test_first_token_immediate_flush()
