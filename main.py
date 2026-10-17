# ==========================================
# imports brain # Imports and Configuration
# ==========================================

import sys, os, shutil, uvicorn
from typing import TypedDict, List
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

# --- 0. IMPORT YOUR CONFIG ---
import config 

# 1. THE "DDGS" SHIM
try:
    import duckduckgo_search
    sys.modules["ddgs"] = duckduckgo_search
except ImportError:
    pass

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END

# ==========================================
# Initialization
# ==========================================

os.makedirs(config.TEMP_DIR, exist_ok=True)
app = FastAPI(title="Elite Free Agentic RAG 0.2.0")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": config.LLM_MODEL, "date": "March 2026"}

embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
llm_endpoint = HuggingFaceEndpoint(
    repo_id=config.LLM_MODEL,
    huggingfacehub_api_token=config.HF_TOKEN,
    temperature=0.1,
    max_new_tokens=512,
    task="conversational"
)
llm = ChatHuggingFace(llm=llm_endpoint)
if config.WEB_SEARCH_PROVIDER == "parallel":
    from parallel_search import ParallelSearchRun
    search_tool = ParallelSearchRun()
    search_source = "Parallel"
elif config.WEB_SEARCH_PROVIDER == "duckduckgo":
    search_tool = DuckDuckGoSearchRun()
    search_source = "DuckDuckGo"
else:
    raise ValueError("WEB_SEARCH_PROVIDER must be 'duckduckgo' or 'parallel'.")

# ==========================================
# Agent Logic (The Brain)
# ==========================================

class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    sources: List[str]
    next_step: str

def supervisor(state: AgentState):
    """FORCED ROUTING: Decides between Local PDF or Global Web."""
    q = state["question"].lower()
    
    # 1. PDF TRIGGER: Only if they explicitly ask about the document/file
    pdf_triggers = ["pdf", "file", "document", "uploaded", "summary", "this about", "content"]
    if any(x in q for x in pdf_triggers):
        return {"next_step": "vector_store"}
    
    # 2. DEFAULT TO WEB: For everything else (especially 2026, prices, news)
    # This ensures "What is the price..." always hits the internet.
    return {"next_step": "web_search"}

def vector_store_node(state: AgentState):
    """Searches the local PDF database."""
    if not os.path.exists(config.DB_PATH): 
        return {"next_step": "generate", "context": [], "sources": []}
    db = FAISS.load_local(config.DB_PATH, embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(state["question"], k=config.TOP_K)
    return {
        "context": [d.page_content for d in docs], 
        "sources": list(set([d.metadata.get("source", "Unknown PDF") for d in docs])), 
        "next_step": "generate"
    }

def web_search_node(state: AgentState):
    """Fetches real-time data from March 2026."""
    try:
        # Preserve DuckDuckGo's existing query; Parallel receives the original question.
        search_query = (
            state["question"] if config.WEB_SEARCH_PROVIDER == "parallel"
            else f"{state['question']} current price March 2026"
        )
        res = search_tool.run(search_query)
        return {"context": [f"REAL-TIME WEB DATA (2026): {res}"], "sources": [search_source], "next_step": "generate"}
    except:
        return {"context": ["Search currently unavailable."], "sources": [], "next_step": "generate"}

def generate_node(state: AgentState):
    """Generates the final response based on the logic used."""
    context_text = "\n".join(state.get("context", []))
    
    # We tell the AI it is CURRENTLY 2026 so it stops apologizing for its training data
    system_instr = (
        "You are an expert AI. Today's date is March 27, 2026. "
        "Use the provided 'REAL-TIME WEB DATA' to answer. "
        "Do NOT mention your knowledge cutoff. If data is provided, trust it as the current truth."
    )
    
    msgs = [
        SystemMessage(content=f"{system_instr}\n\nContext:\n{context_text}"),
        HumanMessage(content=state["question"])
    ]
    response = llm.invoke(msgs) 
    return {"answer": response.content}

# ==========================================
# Graph and API Endpoints
# ==========================================

builder = StateGraph(AgentState) 
builder.add_node("supervisor", supervisor); builder.add_node("vector_store", vector_store_node)
builder.add_node("web_search", web_search_node); builder.add_node("generate", generate_node)

builder.set_entry_point("supervisor") 
builder.add_conditional_edges("supervisor", lambda x: x["next_step"], { 
    "vector_store": "vector_store", "web_search": "web_search", "generate": "generate"
})
builder.add_edge("vector_store", "generate"); builder.add_edge("web_search", "generate"); builder.add_edge("generate", END) 
graph = builder.compile() 

class QueryRequest(BaseModel): 
    question: str
    thread_id: str

@app.post("/upload/") 
async def upload_file(file: UploadFile = File(...)):
    path = os.path.join(config.TEMP_DIR, file.filename)
    with open(path, "wb") as f: shutil.copyfileobj(file.file, f)
    try:
        loader = PyPDFLoader(path); docs = loader.load()
        for d in docs: d.metadata["source"] = file.filename
        splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
        chunks = splitter.split_documents(docs)
        if os.path.exists(config.DB_PATH):
            db = FAISS.load_local(config.DB_PATH, embeddings, allow_dangerous_deserialization=True); db.add_documents(chunks)
        else:
            db = FAISS.from_documents(chunks, embeddings)
        db.save_local(config.DB_PATH) 
        return {"message": f"Successfully indexed {file.filename}"}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(path): os.remove(path)

@app.post("/ask/") 
async def ask_question(request: QueryRequest):
    try:
        inputs = {"question": request.question, "context": [], "sources": [], "answer": "", "next_step": ""}
        result = graph.invoke(inputs) 
        # Crucial: return result["next_step"] as logic_used to see what the agent actually did
        return {"answer": result["answer"], "sources": result.get("sources", []), "logic_used": result.get("next_step")}
    except Exception as e: raise HTTPException(status_code=500, detail=f"Agent Error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
