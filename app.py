"""Flask web app for DocMind AI, with MongoDB-backed user accounts."""
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from functools import lru_cache, wraps

from flask import Flask, abort, flash, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from src.embedding_manager import OllamaError
from src.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.getenv("SECRET_KEY") or secrets.token_hex(32),
    MAX_CONTENT_LENGTH=85 * 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.getenv("COOKIE_SECURE", "false").lower() == "true",
    PERMANENT_SESSION_LIFETIME=timedelta(days=14),
)
ALLOWED_MODES = {"Auto", "PDF Chat", "General AI Chat"}
MAX_PDF_FILES = 4
MAX_PDF_BYTES = 20 * 1024 * 1024


@lru_cache(maxsize=1)
def get_pipeline():
    return RAGPipeline()


@lru_cache(maxsize=1)
def get_users_collection():
    from pymongo import ASCENDING, MongoClient
    from urllib.parse import quote_plus

    database_name = os.getenv("MONGODB_DB", "docmind")
    uri = os.getenv("MONGODB_URI")
    if not uri:
        host = os.getenv("MONGODB_HOST", "127.0.0.1")
        username = os.getenv("MONGODB_USER")
        password = os.getenv("MONGODB_PASSWORD")
        if username and password:
            uri = f"mongodb://{quote_plus(username)}:{quote_plus(password)}@{host}:27017/{database_name}?authSource={database_name}"
        else:
            uri = f"mongodb://{host}:27017/{database_name}"
    client = MongoClient(uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
    client.admin.command("ping")
    users = client[database_name]["users"]
    users.create_index([("email", ASCENDING)], unique=True)
    return users


@app.context_processor
def provide_csrf_token():
    def csrf_token():
        token = session.get("csrf_token")
        if token is None:
            token = secrets.token_urlsafe(32)
            session["csrf_token"] = token
        return token

    return {"csrf_token": csrf_token}


@app.before_request
def validate_csrf():
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        expected = session.get("csrf_token", "")
        supplied = request.form.get("csrf_token", "") or request.headers.get("X-CSRF-Token", "")
        if not expected or not secrets.compare_digest(expected, supplied):
            if request.path.startswith("/api/"):
                return jsonify({"error": "This page expired. Refresh and try again."}), 400
            abort(400)


@app.after_request
def add_security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
        "font-src 'self'; connect-src 'self'; form-action 'self'; frame-ancestors 'none'; base-uri 'self'",
    )
    return response


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            if request.path.startswith("/api/"):
                return jsonify({"error": "Please sign in to continue."}), 401
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped


def mongo_error_message():
    return "MongoDB is unavailable. Start MongoDB, then try again."


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("index"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not email or not password:
            flash("Enter your email address and password.", "error")
        else:
            try:
                user = get_users_collection().find_one({"email": email})
                if user and check_password_hash(user["password_hash"], password):
                    session.clear()
                    session["user_id"] = str(user["_id"])
                    session["email"] = user["email"]
                    if request.form.get("remember") == "on":
                        session.permanent = True
                    return redirect(url_for("index"))
                flash("Email or password is incorrect.", "error")
            except Exception:
                logging.exception("Sign in failed while accessing MongoDB")
                flash(mongo_error_message(), "error")
    return render_template("auth.html", page="login")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if session.get("user_id"):
        return redirect(url_for("index"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        if "@" not in email or len(email) > 254:
            flash("Enter a valid email address.", "error")
        elif len(password) < 10:
            flash("Use a password with at least 10 characters.", "error")
        elif len(password) > 128:
            flash("Use a password with no more than 128 characters.", "error")
        elif password != confirm:
            flash("The passwords do not match.", "error")
        else:
            try:
                from pymongo.errors import DuplicateKeyError

                result = get_users_collection().insert_one({
                    "email": email,
                    "password_hash": generate_password_hash(password, method="scrypt"),
                    "created_at": datetime.now(timezone.utc),
                })
                session.clear()
                session["user_id"] = str(result.inserted_id)
                session["email"] = email
                flash("Your DocMind AI account is ready.", "success")
                return redirect(url_for("index"))
            except DuplicateKeyError:
                flash("An account with that email already exists. Sign in instead.", "error")
            except Exception:
                logging.exception("Account creation failed")
                flash(mongo_error_message(), "error")
    return render_template("auth.html", page="signup")


@app.get("/forgot-password")
def forgot_password():
    return render_template("forgot_password.html")


@app.post("/logout")
@login_required
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.get("/")
@login_required
def index():
    return render_template("index.html", user_email=session.get("email", ""))


@app.get("/api/documents")
@login_required
def documents():
    try:
        return jsonify({"documents": get_pipeline().store.list_documents(owner_id=session["user_id"])})
    except Exception:
        logging.exception("Could not list indexed documents")
        return jsonify({"error": "Could not load the document library."}), 500


@app.post("/api/upload")
@login_required
def upload():
    files = [item for item in request.files.getlist("files") if item.filename]
    if not files:
        return jsonify({"error": "Choose one or more PDF files first."}), 400
    if len(files) > MAX_PDF_FILES:
        return jsonify({"error": f"Select no more than {MAX_PDF_FILES} PDFs at a time."}), 400
    results = []
    for item in files:
        filename = secure_filename(item.filename or "")
        data = item.read(MAX_PDF_BYTES + 1)
        if not filename or not filename.lower().endswith(".pdf"):
            results.append({"filename": item.filename or "Unknown file", "status": "error", "error": "Only PDF files are supported."})
            continue
        if len(data) > MAX_PDF_BYTES:
            results.append({"filename": filename, "status": "error", "error": "Each PDF must be 20 MB or smaller."})
            continue
        try:
            result = get_pipeline().index_pdf(filename, data, owner_id=session["user_id"])
            results.append(result)
        except (ValueError, OllamaError) as exc:
            results.append({"filename": filename, "status": "error", "error": str(exc)})
        except Exception:
            logging.exception("PDF indexing failed for %s", filename)
            results.append({"filename": filename, "status": "error", "error": "Indexing failed. Check Ollama and the app logs."})
    code = 200 if any(row["status"] in {"indexed", "skipped"} for row in results) else 422
    return jsonify({"results": results}), code


@app.post("/api/chat")
@login_required
def chat():
    payload = request.get_json(silent=True) or {}
    question = payload.get("question")
    mode = payload.get("mode", "Auto")
    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "Enter a question to continue."}), 400
    if len(question) > 8000:
        return jsonify({"error": "Keep questions under 8,000 characters."}), 400
    if not isinstance(mode, str) or mode not in ALLOWED_MODES:
        return jsonify({"error": "Choose Auto, PDF Chat, or General AI Chat."}), 400
    try:
        return jsonify(get_pipeline().ask(question.strip(), mode, owner_id=session["user_id"]))
    except OllamaError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:
        logging.exception("Question answering failed")
        return jsonify({"error": "The request failed. Check the local app logs and your Ollama setup."}), 500


@app.delete("/api/documents/<document_id>")
@login_required
def delete_document(document_id):
    try:
        get_pipeline().delete_document(document_id, owner_id=session["user_id"])
        return jsonify({"ok": True})
    except Exception:
        logging.exception("Could not delete document %s", document_id)
        return jsonify({"error": "Could not remove that document."}), 500


@app.errorhandler(400)
def bad_request(_error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Invalid request. Refresh and try again."}), 400
    return render_template("auth.html", page="login"), 400


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "The selected files exceed the 80 MB request limit."}), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False)
