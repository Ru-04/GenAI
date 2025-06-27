import os
import re
import ast
import httpx
import uuid
import logging
import tempfile
import subprocess
from typing import TypedDict, Optional

from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent")

# --- Utility to normalize prompt text for better vector search ---
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())

# Initialize embedding model and vector DB
embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")
chroma_db = Chroma(embedding_function=embedding_model, persist_directory="chroma_memory")

try:
    _ = chroma_db.similarity_search("vector db test", k=1)
    logger.info("[vector_db] Vector DB initialized and responsive.")
except Exception as e:
    logger.error(f"[vector_db] Initialization check failed: {e}")

class CodeState(TypedDict, total=False):
    user_prompt: str
    generated_code: str
    error: Optional[str]
    fixed_code: Optional[str]
    output: Optional[str]

# --- Utility to classify prompt ---
def is_code_prompt(prompt: str) -> bool:
    keywords = ["code", "function", "loop", "array", "string", "python", "program", "compile", "logic", "bug","write","print","debug","algorithm","script","variable","class","method","syntax","error","fix","run","execute"]
    return any(word in prompt.lower() for word in keywords)

# --- Retrieve similar code-related prompts ---
def retrieve_similar_prompts(prompt: str) -> list:
    logger.info("[vector_db] Searching for similar prompts...")
    try:
        prompt = normalize_text(prompt)
        results = chroma_db.similarity_search(prompt, k=5)
        filtered = [
            r for r in results
            if is_code_prompt(r.metadata.get("prompt", ""))
        ]
        logger.info(f"[vector_db] Retrieved {len(filtered)} filtered code prompts.")
        return [r.metadata["prompt"] + "\n" + r.page_content for r in filtered]
    except Exception as e:
        logger.error(f"[vector_db] Retrieval failed: {e}")
        return []

# --- Step 1: Generate code from prompt ---
def generate_code(state: CodeState) -> CodeState:
    prompt = state["user_prompt"]
    logger.info("[generate_code] Prompt: %s", prompt)

    if not is_code_prompt(prompt):
        logger.warning("[generate_code] Non-code prompt received: %s", prompt)
        return {
            **state,
            "generated_code": "",
            "error": " This system is for code generation only. Please ask programming-related questions."
        }

    memory_snippets = retrieve_similar_prompts(prompt)
    memory_context = "\n\n".join(memory_snippets)

    try:
        response = httpx.post(
            "http://127.0.0.1:11434/api/chat",
            json={
                "model": "llama3.2:3b",
                "messages": [
                    {"role": "system", "content": "Use the past examples to assist."},
                    {"role": "user", "content": f"Past Examples:\n{memory_context}\n\nCurrent:\n{prompt}"},
                ],
                "stream": False,
            },
            timeout=180
        )
        code = response.json()["message"]["content"]
        logger.info("[generate_code] Raw response:\n%s", code)

        match = re.search(r"```(?:[\w]*)?\n([\s\S]*?)```", code)
        cleaned_code = match.group(1).strip() if match else code.strip()

        logger.info("[generate_code] Cleaned code:\n%s", cleaned_code)

        if is_code_prompt(prompt):
            chroma_db.add_texts(
                texts=[cleaned_code],
                metadatas=[{"prompt": prompt}],
                ids=[str(uuid.uuid4())]
            )
            logger.info("[vector_db] Code stored in vector DB.")

        return {**state, "generated_code": cleaned_code}

    except Exception as e:
        logger.error("[generate_code] LLaMA call failed: %s", e)
        raise RuntimeError("LLaMA model call failed.")

# --- Step 2: Check syntax ---
def check_syntax(state: CodeState) -> CodeState:
    code = state["generated_code"]
    try:
        ast.parse(code)
        logger.info("[check_syntax] Code is valid.")
        return {**state, "error": None}
    except SyntaxError as e:
        logger.warning("[check_syntax] Syntax error: %s", e)
        return {**state, "error": f"SyntaxError: {e}"}

# --- Step 3: Run raw generated code and catch runtime errors ---
def run_raw_code(state: CodeState) -> CodeState:
    code = state["generated_code"]
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp:
            temp.write(code)
            temp_path = temp.name

        result = subprocess.run(["python", temp_path], capture_output=True, text=True, timeout=10)
        output = result.stdout + result.stderr

        if result.returncode == 0:
            logger.info("[run_raw] Code ran successfully.")
            return {**state, "output": output, "error": None}
        else:
            logger.warning("[run_raw] Runtime error occurred.")
            return {**state, "error": f"RuntimeError: {output}", "output": None}

    except Exception as e:
        logger.error("[run_raw] Exception: %s", e)
        return {**state, "error": f"RuntimeError: {e}", "output": None}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# --- Step 4: Fix code ---
def fix_code(state: CodeState) -> CodeState:
    code = state["generated_code"]
    error = state.get("error", "")
    prompt = f"The following code has an error ({error}):\n\n{code}\n\nPlease fix it."

    try:
        response = httpx.post(
            "http://127.0.0.1:11434/api/chat",
            json={
                "model": "llama3.2:3b",
                "messages": [
                    {"role": "system", "content": "You fix broken code."},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            },
            timeout=180
        )
        fixed = response.json()["message"]["content"]
        logger.info("[fix_code] Raw fix response:\n%s", fixed)

        match = re.search(r"```(?:[\w]*)?\n([\s\S]*?)```", fixed)
        fixed_code = match.group(1).strip() if match else fixed.strip()

        logger.info("[fix_code] Final fixed code:\n%s", fixed_code)
        return {**state, "fixed_code": fixed_code}

    except Exception as e:
        logger.error("[fix_code] LLaMA fix failed: %s", e)
        return {**state, "fixed_code": "", "output": f"Fixing failed: {e}"}

# --- Step 5: Run fixed code ---
def run_code(state: CodeState) -> CodeState:
    code_to_run = state.get("fixed_code")

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp:
            temp.write(code_to_run)
            temp_path = temp.name

        result = subprocess.run(["python", temp_path], capture_output=True, text=True, timeout=10)
        output = result.stdout + result.stderr
        logger.info("[run_code] Code output:\n%s", output)

        return {**state, "output": output}
    except Exception as e:
        logger.error("[run_code] Runtime error: %s", e)
        return {**state, "output": f"Runtime error: {e}"}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# --- LangGraph Agent Definition ---
def build_agent():
    builder = StateGraph(CodeState)

    builder.add_node("generate", RunnableLambda(generate_code))
    builder.add_node("check", RunnableLambda(check_syntax))
    builder.add_node("run_raw", RunnableLambda(run_raw_code))
    builder.add_node("fix", RunnableLambda(fix_code))
    builder.add_node("run", RunnableLambda(run_code))

    builder.set_entry_point("generate")
    builder.add_edge("generate", "check")

    def route_after_check(state: CodeState) -> str:
        return "run_raw" if not state.get("error") else "fix"

    def route_after_run_raw(state: CodeState) -> str:
        return "fix" if state.get("error") else "end"

    builder.add_conditional_edges("check", route_after_check, {
        "run_raw": "run_raw",
        "fix": "fix"
    })

    builder.add_conditional_edges("run_raw", route_after_run_raw, {
    "fix": "fix",
    "end": "__end__"
    })


    builder.add_edge("fix", "run")
    builder.set_finish_point("run_raw")
    builder.set_finish_point("run")

    logger.info("[build_agent] Agent compiled.")
    return builder.compile()