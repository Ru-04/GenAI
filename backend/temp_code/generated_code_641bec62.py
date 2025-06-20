from flask import Flask, request, jsonify

app = Flask(__name__)

# In-memory data store for users
users = {}


def validate_name(name):
    """Validate user name"""
    if not isinstance(name, str) or len(name.strip()) == 0:
        return False
    return True


def validate_email(email):
    """Validate email address"""
    if "@" not in email:
        return False
    return True


def validate_password(password):
    """Validate password length"""
    if len(password.strip()) < 8:
        return False
    return True


@app.route("/users", methods=["POST"])
def create_user():
    data = request.get_json()
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")

    # Input validation
    if not (
        validate_name(name) and validate_email(email) and validate_password(password)
    ):
        return jsonify({"error": "Invalid input"}), 400

    # Create new user
    users[email] = {"name": name, "password": password}

    return jsonify({"message": "User created successfully"}), 201


@app.route("/users/<id>", methods=["GET"])
def get_user(id):
    if id in users:
        return jsonify(users[id])
    else:
        return jsonify({"error": "User not found"}), 404


@app.route("/users/<id>", methods=["PUT"])
def update_user(id):
    user = users.get(id)
    if user is None:
        return jsonify({"error": "User not found"}), 404

    data = request.get_json()
    name = data.get("name")
    email = data.get("email")

    # Input validation
    if validate_name(name) and validate_email(email):
        user["name"] = name
        user["email"] = email

        return jsonify({"message": "User updated successfully"}), 200
    else:
        return jsonify({"error": "Invalid input"}), 400


@app.route("/users/<id>", methods=["DELETE"])
def delete_user(id):
    if id in users:
        del users[id]
        return jsonify({"message": "User deleted successfully"}), 200
    else:
        return jsonify({"error": "User not found"}), 404


if __name__ == "__main__":
    app.run(debug=True)
