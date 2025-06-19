from flask import Flask, request, jsonify
from flask_cors import CORS
import requests as req
import subprocess
import uuid
import os
import ast
import re

app = Flask(__name__)
CORS(app, resources={r"/generate": {"origins": "*"}})

TEMP_DIR = "temp_code"
os.makedirs(TEMP_DIR, exist_ok=True)

# Extract clean Python code from markdown
def clean_code_block(code):
    print("[CLEAN] Cleaning markdown from generated code...")
    match = re.search(r"```python(.*?)```", code, re.DOTALL)
    if match:
        return match.group(1).strip()
    return re.sub(r"```[\w]*\n?|```", "", code).strip()

# Use Ollama to talk to qwen:4b
def get_code_from_qwen(prompt):
    print("[QWEN4B] Sending prompt to Qwen4b...")
    try:
        response = req.post("http://localhost:11434/api/chat", json={
            "model": "qwen:7b",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }, timeout=90)
        response.raise_for_status()
        print("[QWEN4B] Received response.")
        return response.json().get("message", {}).get("content", "")
    except Exception as e:
        print("[ERROR] Qwen4b request failed:", str(e))
        raise

# Validate Python syntax
def is_valid_python_syntax(code):
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, str(e)

# Format using Black
def format_with_black(filename):
    try:
        subprocess.run(["black", filename], check=True, capture_output=True)
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except subprocess.CalledProcessError as e:
        raise Exception(e.stderr.decode())

@app.route('/generate', methods=['POST'])
def generate_and_test():
    print("[API] /generate called")
    user_prompt = request.json.get('prompt')
    if not user_prompt:
        print("[ERROR] No prompt provided.")
        return jsonify({"error": "No prompt provided"}), 400

    file_id = str(uuid.uuid4())[:8]
    filename = os.path.join(TEMP_DIR, f"generated_code_{file_id}.py")
    cleaned_code = ""

    try:
        code = get_code_from_qwen(user_prompt)
        cleaned_code = clean_code_block(code)

        # Retry loop until code is valid
        attempts = 0
        max_attempts = 5
        valid = False
        formatted_code = cleaned_code

        while attempts < max_attempts:
            print(f"[CHECK] Attempt {attempts + 1} of {max_attempts}")
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(cleaned_code)

            valid, syntax_error = is_valid_python_syntax(cleaned_code)
            if valid:
                print("[VALID] Code is syntactically correct.")
                break

            print("[FIX] Code has syntax errors. Re-prompting Qwen to fix...")
            fix_prompt = f"Fix the following Python code with formatting or syntax errors:\n\n{cleaned_code}\n\nOnly return the corrected code in markdown."
            cleaned_code = clean_code_block(get_code_from_qwen(fix_prompt))
            attempts += 1

        if not valid:
            return jsonify({"code": cleaned_code, "output": "", "error": syntax_error}), 400

        # Now apply Black formatting AFTER the code is correct
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(cleaned_code)
            formatted_code = format_with_black(filename)
        except Exception as e:
            print("[BLACK] Formatting failed:", str(e))
            formatted_code = cleaned_code  # fallback to original if formatting fails

        print("[RUN] Running valid Python code...")
        run_result = subprocess.run(["python", filename], capture_output=True, text=True, timeout=10)

        output = run_result.stdout.strip()
        error = run_result.stderr.strip()

        print("[OUTPUT]", output if output else "<no output>")
        print("[ERROR]", error if error else "<no error>")

        os.remove(filename)
        print("[DONE] Code executed.")

        return jsonify({
            "code": formatted_code,
            "success": run_result.returncode == 0,
            "output": output,
            "error": error
        })

    except subprocess.TimeoutExpired:
        return jsonify({"code": cleaned_code, "output": "", "error": "Execution timed out"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=False)
