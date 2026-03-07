
import asyncio
import os
import sys
import time
from pathlib import Path

# Add src to sys.path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.session import SessionManager

async def verify_highlight_logic():
    session_id = "test_logic_session"
    SessionManager.init_session(session_id)
    
    print(f"--- Verification: Highlight Isolation Logic ---")
    
    # 1. Simulate Question 1 and Answer 1 with its own highlights
    print("\n1. Simulating Question 1...")
    annos1 = [
        {"page": 1, "x": 10, "y": 20, "width": 100, "height": 20, "color": "red"},
        {"page": 1, "x": 10, "y": 50, "width": 150, "height": 20, "color": "red"}
    ]
    
    SessionManager.add_message(
        role="assistant",
        content="Answer 1 related to page 1.",
        msg_type="answer",
        annotations=annos1 # Store in message 1
    )
    print(f"   - Message 1 stored with {len(annos1)} annotations.")
    
    # 2. Simulate Question 2 and Answer 2 with different highlights
    print("\n2. Simulating Question 2...")
    annos2 = [
        {"page": 5, "x": 30, "y": 100, "width": 200, "height": 30, "color": "blue"}
    ]
    
    SessionManager.add_message(
        role="assistant",
        content="Answer 2 related to page 5.",
        msg_type="answer",
        annotations=annos2 # Store in message 2
    )
    print(f"   - Message 2 stored with {len(annos2)} annotations.")
    
    # 3. Final Verification
    messages = SessionManager.get_messages(session_id=session_id)
    # Filter only assistant answers
    assistant_msgs = [m for m in messages if m.get("role") == "assistant" and m.get("msg_type") == "answer"]
    
    print("\n--- Logic Results ---")
    print(f"Total assistant messages found: {len(assistant_msgs)}")
    
    for i, msg in enumerate(assistant_msgs):
        m_annos = msg.get("annotations", [])
        m_pages = [a["page"] for a in m_annos]
        print(f"Message {i+1} (Page {m_pages}): {len(m_annos)} annotations stored.")
        
    # Validation Points
    if len(assistant_msgs) == 2:
        res1 = assistant_msgs[0]["annotations"]
        res2 = assistant_msgs[1]["annotations"]
        
        # Point A: Different contents
        is_different = (res1 != res2)
        # Point B: Different memory objects
        is_isolated = (id(res1) != id(res2))
        
        print(f"\n[Verification Outcome]")
        print(f"- Content Differentiation: {'SUCCESS' if is_different else 'FAILED'}")
        print(f"- Memory Isolation: {'SUCCESS' if is_isolated else 'FAILED'}")
        
        if is_different and is_isolated:
            print("\n>> Conclusion: Highlight isolation is correctly implemented.")
            print(">> Each message now owns its specific highlight state.")
        else:
            print("\n>> Conclusion: Isolation logic failure.")
    else:
        print("\n>> Error: Could not find both messages in history.")

if __name__ == "__main__":
    asyncio.run(verify_highlight_logic())
