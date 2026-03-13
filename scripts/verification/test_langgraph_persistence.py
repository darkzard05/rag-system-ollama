import asyncio
import operator
from typing import Annotated, TypedDict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

class MockState(TypedDict):
    input: str
    counter: Annotated[int, operator.add]

def node_a(state: MockState) -> dict[str, Any]:
    print(f"Node A: input={state['input']}, counter={state.get('counter', 0)}")
    return {"counter": 1}

async def verify_persistence():
    workflow = StateGraph(MockState)
    workflow.add_node("node_a", node_a)
    workflow.add_edge(START, "node_a")
    workflow.add_edge("node_a", END)
    
    memory = InMemorySaver()
    graph = workflow.compile(checkpointer=memory)
    
    config = {"configurable": {"thread_id": "test_session"}}
    
    print("\n--- First Call ---")
    # counter starts at 0, should become 1
    res1 = await graph.ainvoke({"input": "Hello", "counter": 0}, config=config)
    print(f"Result 1 Counter: {res1['counter']}")
    
    print("\n--- Second Call (Same thread_id) ---")
    # counter starts from previous 1, should become 2
    # input is new, but counter is managed by reducer + checkpointer
    res2 = await graph.ainvoke({"input": "World"}, config=config)
    print(f"Result 2 Counter: {res2['counter']}")
    
    if res2['counter'] == 2:
        print("\n✅ SUCCESS: State persisted across calls with same thread_id!")
    else:
        print("\n❌ FAILURE: State was lost or reset.")

if __name__ == "__main__":
    asyncio.run(verify_persistence())
