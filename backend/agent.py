import os
import httpx
import logging
import tempfile
import subprocess
from typing import TypedDict, Optional, Annotated
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph
import py_compile
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent")

# --- Utility ---
def clean_markdown_wrappers(text: str) -> str:
    # Removes all markdown code block wrappers and language specifiers
    return re.sub(r"(```[a-z]*\n?|```)", "", text.strip(), flags=re.IGNORECASE)

def extract_first_code_block(text: str) -> str:
    # Extract content between first ```python and ``` if they exist
    match = re.search(r"```(?:python)?\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

# --- Code state definition ---
class CodeState(TypedDict, total=False):
    user_prompt: Annotated[str, "static"]
    generated_code: str
    test_code: str
    error: Optional[str]
    structured_issues: Optional[str]
    fixed_code: Optional[str]
    output: Optional[str]
    syntax_attempts: int
    test_regen_attempts: int

# --- Step 1: Plan and generate clean code only ---
def plan_strategy_generate_code(state: CodeState) -> CodeState:
    prompt = state["user_prompt"]
    logger.info("[plan_strategy_generate_code] Prompt received:\n%s", prompt)

    instruction = f"""
You are a professional Python developer. Implement exactly what is requested below.
Return ONLY the raw Python code with NO additional text, comments, or explanations.

Request: {prompt}

Example for "print hello":
print('hello')

Example for "sum of natural numbers":
def sum_natural(n):
    return n * (n + 1) // 2
"""
    try:
        response = httpx.post(
            "http://127.0.0.1:11434/api/chat",
            json={
                "model": "llama3.2:3b",  # Using llama3 instead of deepseek
                "messages": [
                    {
                        "role": "system", 
                        "content": "You output exactly one raw Python solution. No text, no explanations, no markdown. Just pure code."
                    },
                    {"role": "user", "content": instruction},
                ],
                "stream": False,
                "options": {
                    "temperature": 0.2,  # Lower temperature for more deterministic output
                    "num_ctx": 2048
                }
            },
            timeout=500
        )
        raw_content = response.json()["message"]["content"]
        
        # Enhanced cleaning
        code = extract_first_code_block(raw_content)
        code = clean_markdown_wrappers(code)
        code = re.sub(r'^.*?(?=(def |class |print |import |from |[a-zA-Z_][a-zA-Z0-9_]*\s*=))', '', code, flags=re.DOTALL)
        code = code.strip()
        
        logger.info("[plan_strategy_generate_code] Code generated:\n%s", code)
        return {
            **state,
            "generated_code": code,
            "syntax_attempts": 0,
            "test_regen_attempts": 0,
        }
    except Exception as e:
        logger.error("[plan_strategy_generate_code] Failed: %s", e)
        return {**state, "generated_code": "# Code generation failed."}

# --- Step 2: Syntax validation using py_compile ---
def validate_syntax(state: CodeState) -> CodeState:
    code = state["generated_code"]
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp:
            temp.write(code)
            temp_path = temp.name
        py_compile.compile(temp_path, doraise=True)
        logger.info("[validate_syntax] Syntax is valid.")
        return {**state, "error": None}
    except py_compile.PyCompileError as e:
        logger.warning("[validate_syntax] Syntax error detected. Attempt: %d", state.get("syntax_attempts", 0) + 1)
        logger.warning("[validate_syntax] Error details: %s", str(e))
        return {
            **state,
            "error": str(e),
            "syntax_attempts": state.get("syntax_attempts", 0) + 1,
            "structured_issues": str(e)
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# --- Step 3: Fix code ---
def fix_code(state: CodeState) -> CodeState:
    if state.get("syntax_attempts", 0) >= 3:
        return {**state, "error": "Max syntax fix attempts reached"}
    
    code = state["generated_code"]
    error = state.get("structured_issues") or state.get("error") or "Unknown error"
    logger.info("[fix_code] Fixing based on error: %s", error)
    
    prompt = f"""
This Python code has an error:
{error}

Fix ONLY the code below and return JUST the corrected code with NO other text:

{code}
"""
    try:
        response = httpx.post(
            "http://127.0.0.1:11434/api/chat",
            json={
                "model": "llama3.2:3b",
                "messages": [
                    {"role": "system", "content": "You fix Python code errors. Return ONLY the corrected code with NO explanations."},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": 0.1  # Very low temperature for fixes
                }
            },
            timeout=500
        )
        fixed_code = extract_first_code_block(response.json()["message"]["content"])
        fixed_code = clean_markdown_wrappers(fixed_code).strip()
        
        logger.info("[fix_code] Code fixed. Attempt: %d", state.get("syntax_attempts", 0))
        return {
            **state,
            "fixed_code": fixed_code,
            "generated_code": fixed_code,
            "error": None
        }
    except Exception as e:
        logger.error("[fix_code] Fixing failed: %s", e)
        return {**state, "fixed_code": "", "error": f"Fixing failed: {e}"}

# --- Step 4: Generate test cases (enhanced version) ---
def check(state: CodeState) -> CodeState:
    code = state["generated_code"]
    prompt = f"""
Write comprehensive Python test cases for the following code.
Include both positive and negative test cases.
Return ONLY the test code with NO explanations or additional text.
Include print statements showing expected vs actual results.

Code to test:
{code}
"""
    try:
        response = httpx.post(
            "http://127.0.0.1:11434/api/chat",
            json={
                "model": "llama3.2:3b",
                "messages": [
                    {"role": "system", "content": "You write detailed Python test code with print statements showing expected vs actual results."},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": 0.3
                }
            },
            timeout=500
        )
        raw_test_content = response.json()["message"]["content"]
        
        test_code = extract_first_code_block(raw_test_content)
        test_code = clean_markdown_wrappers(test_code).strip()
        
        # Extract and store only the relevant test cases
        test_cases = []
        current_test = None
        for line in test_code.split('\n'):
            line = line.strip()
            if line.startswith('def test_'):
                if current_test:
                    test_cases.append(current_test)
                current_test = {"name": line, "assertions": []}
            elif line.startswith('assert ') or line.startswith('print('):
                if current_test:
                    current_test["assertions"].append(line)
        
        if current_test:
            test_cases.append(current_test)
        
        logger.info("[check] Extracted test cases:")
        for test in test_cases:
            logger.info(f"Test: {test['name']}")
            for assertion in test['assertions']:
                logger.info(f"  - {assertion}")
        
        return {
            **state,
            "test_code": test_code,
            "test_cases": test_cases,  # Store structured test cases
            "test_regen_attempts": state.get("test_regen_attempts", 0) + 1
        }
    except Exception as e:
        logger.error("[check] Test generation failed: %s", e)
        return {**state, "test_code": "# Test generation failed."}

# --- Step 5: Run test cases (enhanced version) ---
def run_tests(state: CodeState) -> CodeState:
    code = state["generated_code"]
    tests = state["test_code"]
    combined = f"{code}\n\n{tests}"
    temp_path = None
    
    logger.info("[run_tests] Running tests for code:\n%s", code)
    logger.info("[run_tests] Using these test cases:\n%s", tests)
    
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp:
            temp.write(combined)
            temp_path = temp.name
        
        result = subprocess.run(["python", temp_path], capture_output=True, text=True, timeout=20)  # Increased timeout
        
        output = result.stdout + result.stderr
        logger.info("[run_tests] Test execution output:\n%s", output)
        
        # Parse test results
        test_results = []
        for line in output.split('\n'):
            if "Expected:" in line or "Actual:" in line or "Test:" in line or "FAILED" in line:
                test_results.append(line.strip())
        
        if result.returncode == 0:
            logger.info("[run_tests] ALL TESTS PASSED")
            return {
                **state,
                "error": None,
                "output": output,
                "test_results": test_results,  # Store detailed results
                "test_status": "PASSED"
            }
        else:
            logger.warning("[run_tests] SOME TESTS FAILED")
            failure_lines = [line for line in output.split('\n') if 'FAIL' in line or 'Error' in line]
            if failure_lines:
                logger.warning("[run_tests] Test failures detected:")
                for failure in failure_lines:
                    logger.warning("[run_tests] %s", failure)
            
            return {
                **state,
                "error": f"TestFailure: {output}",
                "output": output,
                "test_results": test_results,  # Store detailed results
                "test_status": "FAILED"
            }
    except Exception as e:
        logger.error("[run_tests] Error during test run: %s", e)
        return {
            **state,
            "error": f"TestFailure: {e}",
            "test_status": "ERROR"
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# --- Step 6: Final run with enhanced output ---
def run_code(state: CodeState) -> CodeState:
    code = state.get("fixed_code") or state.get("generated_code")
    temp_path = None
    
    # Prepare final output with test details
    final_output = []
    
    # Add test cases if available (formatted nicely)
    if state.get("test_cases"):
        final_output.append("\n=== TEST CASES EXECUTED ===\n")
        for test in state["test_cases"]:
            final_output.append(f"\nTest Function: {test['name']}")
            for assertion in test['assertions']:
                final_output.append(f"  - {assertion}")
    
    # Add test results if available (formatted)
    if state.get("test_results"):
        final_output.append("\n\n=== TEST RESULTS ===\n")
        final_output.extend(state["test_results"])
    
    # Add execution output if available
    if state.get("output"):
        final_output.append("\n=== PROGRAM OUTPUT ===\n")
        final_output.append(state["output"])
    
    # Add status summary
    final_output.append("\n=== FINAL STATUS ===\n")
    if state.get("test_status"):
        final_output.append(f"Test Status: {state['test_status']}")
    if state.get("error"):
        final_output.append(f"\nError Details: {state['error']}")
    else:
        final_output.append("\nAll tests completed successfully")
    
    # Combine all output
    formatted_output = "\n".join(final_output)
    
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp:
            temp.write(code)
            temp_path = temp.name
        
        result = subprocess.run(["python", temp_path], capture_output=True, text=True, timeout=20)
        runtime_output = result.stdout + result.stderr
        
        if result.returncode == 0:
            logger.info("[run_code] Final run successful")
            formatted_output += f"\n\n=== RUNTIME OUTPUT ===\n{runtime_output}"
        else:
            logger.error("[run_code] Runtime error occurred")
            formatted_output += f"\n\n=== RUNTIME ERROR ===\n{runtime_output}"
        
        return {**state, "output": formatted_output}
    except Exception as e:
        logger.error("[run_code] Final run error: %s", e)
        formatted_output += f"\n\n=== CRITICAL ERROR ===\n{str(e)}"
        return {**state, "output": formatted_output}
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# --- Agent flow ---
def build_agent():
    builder = StateGraph(CodeState)
    
    # Add nodes
    builder.add_node("plan_generate", RunnableLambda(plan_strategy_generate_code))
    builder.add_node("validate_syntax", RunnableLambda(validate_syntax))
    builder.add_node("fix_code", RunnableLambda(fix_code))
    builder.add_node("check", RunnableLambda(check))
    builder.add_node("run_tests", RunnableLambda(run_tests))
    builder.add_node("final_run", RunnableLambda(run_code))

    # Set entry point
    builder.set_entry_point("plan_generate")
    builder.add_edge("plan_generate", "validate_syntax")

    # Conditional edges
    def syntax_decider(state: CodeState):
        if state.get("syntax_attempts", 0) >= 3:
            return "final_run"  # Give up after 3 attempts
        return "check" if not state.get("error") else "fix_code"

    def test_decider(state: CodeState):
        if state.get("test_regen_attempts", 0) >= 3:
            return "final_run"  # Give up after 3 attempts
        return "final_run" if not state.get("error") else "fix_code"

    builder.add_conditional_edges("validate_syntax", syntax_decider, {
        "check": "check",
        "fix_code": "fix_code",
        "final_run": "final_run"
    })

    builder.add_edge("fix_code", "validate_syntax")
    builder.add_edge("check", "run_tests")
    
    builder.add_conditional_edges("run_tests", test_decider, {
        "final_run": "final_run",
        "fix_code": "fix_code"
    })

    # Set finish point
    builder.set_finish_point("final_run")
    logger.info("[build_agent] Agent graph compiled successfully")
    return builder.compile()