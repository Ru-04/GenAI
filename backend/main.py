from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from agent import build_agent, CodeState

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        user_prompt = data.get("prompt", "").strip()

        logger.info("Received prompt: %s", user_prompt)

        state = CodeState({"user_prompt": user_prompt})
        agent = build_agent()
        final_state = agent.invoke(state)

        output = final_state.get("output", "")
        cleaned_code = final_state.get("generated_code", "")
        error = final_state.get("error", "")

        return jsonify({
            "output": output,
            "code": cleaned_code,
            "error": error
        })
    except Exception as e:
        logging.error(f"Agent failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
