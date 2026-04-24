from typing import Literal

from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.state import SupportState
from app.hitl import escalate_to_human
from app.config import PERSIST_DIR, COLLECTION_NAME, TOP_K, CONFIDENCE_THRESHOLD


emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

store = Chroma(
    persist_directory=PERSIST_DIR,
    collection_name=COLLECTION_NAME,
    embedding_function=emb
)

retriever = store.as_retriever(search_kwargs={"k": TOP_K})


def detect_intent(q: str) -> str:
    q = q.lower()
    if any(x in q for x in ["legal", "complaint", "lawsuit", "fraud", "abuse"]):
        return "sensitive"
    return "knowledge"

def process_node(state: SupportState) -> SupportState:
    q = state["query"]
    intent = detect_intent(q)

    docs = retriever.invoke(q)
    chunks = [d.page_content for d in docs]

    if len(chunks) == 0:
        conf = 0.25
    else:
        conf = min(0.9, 0.45 + 0.1 * len(chunks))

    if intent == "sensitive":
        conf = min(conf, 0.45)

    if chunks and conf >= CONFIDENCE_THRESHOLD:
        answer = "Grounded answer:\n" + "\n\n".join(ch[:350] for ch in chunks[:2])
    else:
        answer = "Insufficient confidence to answer automatically."

    state.update({
        "intent": intent,
        "retrieved_chunks": chunks,
        "confidence": conf,
        "answer": answer,
        "escalated": False
    })

    return state


def route_after_process(state: SupportState) -> Literal["output", "hitl"]:
    if state.get("confidence", 0) < CONFIDENCE_THRESHOLD or not state.get("retrieved_chunks"):
        return "hitl"
    return "output"


def output_node(state: SupportState) -> SupportState:
    return state


b = StateGraph(SupportState)
b.add_node("process", process_node)
b.add_node("hitl", escalate_to_human)
b.add_node("output", output_node)

b.set_entry_point("process")
b.add_conditional_edges(
    "process",
    route_after_process,
    {
        "output": "output",
        "hitl": "hitl"
    }
)
b.add_edge("hitl", "output")
b.add_edge("output", END)

graph = b.compile()


if __name__ == "__main__":
    print("Customer Support Assistant is ready. Type 'exit' to quit.\n")

    while True:
        query = input("Ask your question: ").strip()

        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        initial_state = {
            "query": query,
            "intent": "",
            "retrieved_chunks": [],
            "confidence": 0.0,
            "answer": "",
            "escalated": False,
            "human_response": None,
            "citations": []
        }

        result = graph.invoke(initial_state)

        print("\nResponse:")
        print(result["answer"])
        print(f"\nIntent: {result['intent']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Escalated: {result['escalated']}")
        print("-" * 50)
