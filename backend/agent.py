import os
import re
import ast
import httpx
import logging
import tempfile
import subprocess
from typing import TypedDict, Optional

from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent")

# Define state schema
class CodeState(TypedDict, total=False):
    user_prompt: str
    generated_code: str
    error: Optional[str]
    fixed_code: Optional[str]
    output: Optional[str]

# Step 1: Generate code using LLaMA
def generate_code(state: CodeState) -> CodeState:
    prompt = state["user_prompt"]
    logger.info("[generate_code] Prompt: %s", prompt)

    try:
        response = httpx.post(
            "http://127.0.0.1:11434/api/chat",
            json={
                "model": "llama3.2:3b",
                "messages": [
                    {"role": "system", "content": "You are a helpful coding assistant."},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            },
            timeout=180
        )
        code = response.json()["message"]["content"]
        cleaned_code = re.sub(r"```[\w]*\n?([\s\S]*?)```", r"\1", code).strip()

        logger.info("[generate_code] Code generated and cleaned.")
        return {**state, "generated_code": cleaned_code}

    except Exception as e:
        logger.error("[generate_code] LLaMA call failed: %s", e)
        raise RuntimeError("LLaMA model call failed.")

# Step 2: Syntax validation
def check_syntax(state: CodeState) -> CodeState:
    code = state["generated_code"]
    try:
        ast.parse(code)
        logger.info("[check_syntax] Code is valid.")
        return {**state, "error": None}
    except SyntaxError as e:
        logger.warning("[check_syntax] Syntax error: %s", e)
        return {**state, "error": str(e)}

# Step 3: Fix invalid code using LLaMA
def fix_code(state: CodeState) -> CodeState:
    code = state["generated_code"]
    error = state.get("error", "")
    prompt = f"The following code has a syntax error: {error}\n\n{code}\n\nPlease fix it."

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
        fixed_code = re.sub(r"```[\w]*\n?([\s\S]*?)```", r"\1", fixed).strip()

        logger.info("[fix_code] Code fixed.")
        return {**state, "fixed_code": fixed_code}

    except Exception as e:
        logger.error("[fix_code] LLaMA fix failed: %s", e)
        return {**state, "fixed_code": "", "output": f"Fixing failed: {e}"}

# Step 4: Run the code
def run_code(state: CodeState) -> CodeState:
    code_to_run = state.get("fixed_code") or state.get("generated_code")

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp:
            temp.write(code_to_run)
            temp_path = temp.name

        result = subprocess.run(["python", temp_path], capture_output=True, text=True, timeout=10)
        output = result.stdout + result.stderr

        return {**state, "output": output}
    except Exception as e:
        return {**state, "output": f"Runtime error: {e}"}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
# Build LangGraph agent
def build_agent():
    builder = StateGraph(CodeState)

    builder.add_node("generate", RunnableLambda(generate_code))
    builder.add_node("check", RunnableLambda(check_syntax))
    builder.add_node("fix", RunnableLambda(fix_code))
    builder.add_node("run", RunnableLambda(run_code))

    builder.set_entry_point("generate")
    builder.add_edge("generate", "check")

    def routing_fn(state: CodeState) -> str:
        return "fix" if state.get("error") else "run"

    builder.add_conditional_edges("check", routing_fn, {
        "fix": "fix",
        "run": "run"
    })

    builder.add_edge("fix", "run")
    builder.set_finish_point("run")

    logger.info("[build_agent] Agent compiled.")
    return builder.compile()
