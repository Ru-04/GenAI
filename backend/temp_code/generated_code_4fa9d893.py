from flask import Flask, jsonify, request

app = Flask(__name__)

# In-memory dictionary to store tasks
tasks = {}


@app.route("/tasks", methods=["GET"])
def get_all_tasks():
    """Return all tasks"""
    return jsonify(tasks)


@app.route("/tasks/<int:task_id>", methods=["GET"])
def get_task(task_id):
    """Return a specific task by ID"""
    if task_id in tasks:
        return jsonify(tasks[task_id])
    else:
        return jsonify({"error": "Task not found"}), 404


@app.route("/tasks", methods=["POST"])
def create_task():
    """Create a new task"""
    new_task = {
        "id": len(tasks) + 1,
        "title": request.json["title"],
        "description": request.json.get("description", ""),
        "status": "pending",
    }
    tasks[new_task["id"]] = new_task
    return jsonify(new_task), 201


@app.route("/tasks/<int:task_id>", methods=["PUT"])
def update_task(task_id):
    """Update a task by ID"""
    if task_id in tasks:
        tasks[task_id]["title"] = request.json.get("title", tasks[task_id]["title"])
        tasks[task_id]["description"] = request.json.get(
            "description", tasks[task_id]["description"]
        )
        tasks[task_id]["status"] = request.json.get("status", tasks[task_id]["status"])
        return jsonify(tasks[task_id])
    else:
        return jsonify({"error": "Task not found"}), 404


@app.route("/tasks/<int:task_id>", methods=["DELETE"])
def delete_task(task_id):
    """Delete a task by ID"""
    if task_id in tasks:
        del tasks[task_id]
        return jsonify({"message": "Task deleted"})
    else:
        return jsonify({"error": "Task not found"}), 404


if __name__ == "__main__":
    app.run(debug=True)
