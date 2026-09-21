import os
import json
import sqlite3
import uuid
from datetime import datetime

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    session
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from dotenv import load_dotenv

from langchain_groq import ChatGroq

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.load_and_extract_text import (
    extract_text_from_pdf,
    extract_pdf_sections
)

from src.get_summary import generate_detailed_summary


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = os.getenv(
    "UPLOAD_FOLDER",
    "uploads"
)

app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


# ============================================================
# SECRET KEY
# ============================================================

flask_secret_key = os.getenv("FLASK_SECRET_KEY")

if not flask_secret_key:
    flask_secret_key = "research-paper-analyzer-development-key"

app.secret_key = flask_secret_key


# ============================================================
# SESSION CONFIGURATION
# ============================================================

cookie_secure = (
    os.getenv("COOKIE_SECURE", "false").lower() == "true"
)

app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = cookie_secure


# ============================================================
# UPLOAD DIRECTORY
# ============================================================

os.makedirs(
    app.config["UPLOAD_FOLDER"],
    exist_ok=True
)


# ============================================================
# DATABASE
# ============================================================

DATABASE = os.getenv(
    "DATABASE_PATH",
    "research_paper_analyzer.db"
)


def get_db_connection():

    connection = sqlite3.connect(
        DATABASE,
        timeout=30
    )

    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():

    connection = get_db_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            extracted_text TEXT NOT NULL,
            sections TEXT NOT NULL,
            uploaded_at TEXT NOT NULL,
            FOREIGN KEY (user_id)
            REFERENCES users(id)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id INTEGER NOT NULL,
            topic TEXT NOT NULL,
            summary TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (paper_id)
            REFERENCES papers(id)
        )
        """
    )

    connection.commit()
    connection.close()


initialize_database()


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

groq_api_key = os.getenv("GROQ_API_KEY")
llm_model = os.getenv("LLM_MODEL")


# ============================================================
# LAZY LLM INITIALIZATION
# ============================================================

llm = None


def get_llm():

    global llm

    if llm is None:

        if not groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not configured."
            )

        if not llm_model:
            raise RuntimeError(
                "LLM_MODEL is not configured."
            )

        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=llm_model,
            temperature=0
        )

    return llm


# ============================================================
# LIGHTWEIGHT IN-MEMORY RETRIEVAL
# ============================================================

vector_databases = {}


def create_lightweight_vector_db(text):

    chunk_size = 1200
    overlap = 200

    chunks = []

    start = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text)
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    if not chunks:
        return None

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    matrix = vectorizer.fit_transform(chunks)

    return {
        "chunks": chunks,
        "vectorizer": vectorizer,
        "matrix": matrix
    }


def retrieve_relevant_chunks(
    vector_db,
    question,
    top_k=4
):

    if not vector_db:
        return []

    vectorizer = vector_db["vectorizer"]
    matrix = vector_db["matrix"]
    chunks = vector_db["chunks"]

    question_vector = vectorizer.transform(
        [question]
    )

    similarities = cosine_similarity(
        question_vector,
        matrix
    ).flatten()

    ranked_indexes = similarities.argsort()[::-1]

    results = []

    for index in ranked_indexes[:top_k]:

        if similarities[index] <= 0:
            continue

        results.append(
            chunks[index]
        )

    return results


# ============================================================
# USER HELPER
# ============================================================

def get_current_user():

    user_id = session.get("user_id")

    if not user_id:
        return None

    connection = get_db_connection()

    user = connection.execute(
        """
        SELECT id, username, created_at
        FROM users
        WHERE id = ?
        """,
        (user_id,)
    ).fetchone()

    connection.close()

    if user is None:

        session.clear()

        return None

    return user


def login_required():

    return get_current_user() is not None


def get_current_paper():

    paper_id = session.get("paper_id")
    user_id = session.get("user_id")

    if not paper_id or not user_id:
        return None

    connection = get_db_connection()

    paper = connection.execute(
        """
        SELECT *
        FROM papers
        WHERE id = ?
        AND user_id = ?
        """,
        (
            paper_id,
            user_id
        )
    ).fetchone()

    connection.close()

    return paper


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    user = get_current_user()

    if user is None:
        return redirect(
            url_for("login")
        )

    return render_template(
        "index.html",
        username=user["username"]
    )


# ============================================================
# REGISTER
# ============================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if login_required():

        return redirect(
            url_for("index")
        )

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        ).strip()

        confirm_password = request.form.get(
            "confirm_password",
            ""
        ).strip()

        if not username or not password:

            return render_template(
                "register.html",
                error="Username and password are required."
            )

        if len(username) < 3:

            return render_template(
                "register.html",
                error="Username must contain at least 3 characters."
            )

        if len(password) < 6:

            return render_template(
                "register.html",
                error="Password must contain at least 6 characters."
            )

        if password != confirm_password:

            return render_template(
                "register.html",
                error="Passwords do not match."
            )

        password_hash = generate_password_hash(
            password
        )

        connection = get_db_connection()

        try:

            connection.execute(
                """
                INSERT INTO users
                (
                    username,
                    password_hash,
                    created_at
                )
                VALUES (?, ?, ?)
                """,
                (
                    username,
                    password_hash,
                    datetime.now().isoformat()
                )
            )

            connection.commit()

        except sqlite3.IntegrityError:

            connection.close()

            return render_template(
                "register.html",
                error="Username already exists."
            )

        connection.close()

        return redirect(
            url_for("login")
        )

    return render_template(
        "register.html"
    )


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if login_required():

        return redirect(
            url_for("index")
        )

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        ).strip()

        if not username or not password:

            return render_template(
                "login.html",
                error="Please enter username and password."
            )

        connection = get_db_connection()

        user = connection.execute(
            """
            SELECT *
            FROM users
            WHERE username = ?
            """,
            (username,)
        ).fetchone()

        connection.close()

        if user and check_password_hash(
            user["password_hash"],
            password
        ):

            session.clear()

            session["user_id"] = user["id"]
            session["username"] = user["username"]

            return redirect(
                url_for("index")
            )

        return render_template(
            "login.html",
            error="Invalid username or password."
        )

    return render_template(
        "login.html"
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("login")
    )


# ============================================================
# HISTORY
# ============================================================

@app.route("/history")
def history():

    if not login_required():

        return redirect(
            url_for("login")
        )

    user_id = session["user_id"]

    connection = get_db_connection()

    papers = connection.execute(
        """
        SELECT
            id,
            filename,
            uploaded_at
        FROM papers
        WHERE user_id = ?
        ORDER BY uploaded_at DESC
        """,
        (user_id,)
    ).fetchall()

    connection.close()

    return render_template(
        "history.html",
        papers=papers,
        username=session.get("username")
    )


# ============================================================
# VIEW HISTORY
# ============================================================

@app.route(
    "/history/<int:paper_id>"
)
def view_history(paper_id):

    if not login_required():

        return redirect(
            url_for("login")
        )

    user_id = session["user_id"]

    connection = get_db_connection()

    paper = connection.execute(
        """
        SELECT *
        FROM papers
        WHERE id = ?
        AND user_id = ?
        """,
        (
            paper_id,
            user_id
        )
    ).fetchone()

    if not paper:

        connection.close()

        return render_template(
            "history.html",
            papers=[],
            username=session.get("username"),
            error="Research paper not found."
        )

    summaries = connection.execute(
        """
        SELECT
            topic,
            summary,
            created_at
        FROM summaries
        WHERE paper_id = ?
        ORDER BY created_at DESC
        """,
        (paper_id,)
    ).fetchall()

    connection.close()

    sections = json.loads(
        paper["sections"]
    )

    return render_template(
        "history_detail.html",
        paper=paper,
        sections=sections,
        summaries=summaries,
        username=session.get("username")
    )


# ============================================================
# UPLOAD PDF
# ============================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload_pdf():

    if not login_required():

        return jsonify({
            "error": "Please login first."
        }), 401

    user_id = session["user_id"]

    file = request.files.get("file")

    if file is None:

        return jsonify({
            "error": "No file uploaded."
        }), 400

    if not file.filename:

        return jsonify({
            "error": "No file selected."
        }), 400

    if not file.filename.lower().endswith(".pdf"):

        return jsonify({
            "error": "Only PDF files are allowed."
        }), 400

    original_filename = file.filename

    unique_filename = (
        str(uuid.uuid4()) +
        ".pdf"
    )

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        unique_filename
    )

    try:

        file.save(filepath)

        # ----------------------------------------------------
        # EXTRACT PDF TEXT
        # ----------------------------------------------------

        extracted_text = extract_text_from_pdf(
            filepath
        )

        if not extracted_text or not extracted_text.strip():

            return jsonify({
                "error": "Could not extract text from this PDF."
            }), 400

        # ----------------------------------------------------
        # DETECT SECTIONS
        # ----------------------------------------------------

        extracted_sections = extract_pdf_sections(
            full_text=extracted_text
        )

        # ----------------------------------------------------
        # NO EXTRA LLM CALL DURING UPLOAD
        # This makes upload faster and more reliable.
        # ----------------------------------------------------

        if extracted_sections:

            extracted_sections = sorted(
                extracted_sections,
                key=lambda x: x["start"]
            )

            section_with_content = {}

            for i, section in enumerate(
                extracted_sections
            ):

                start = section["start"]

                if i + 1 < len(extracted_sections):

                    end = extracted_sections[
                        i + 1
                    ]["start"]

                else:

                    end = len(extracted_text)

                content = extracted_text[
                    start:end
                ].strip()

                if not content:
                    continue

                section_name = section.get(
                    "section",
                    "Untitled Section"
                )

                subsection_name = section.get(
                    "subsection"
                )

                if subsection_name:

                    key = subsection_name

                else:

                    key = section_name

                section_with_content[key] = content

        else:

            section_with_content = {
                "Full Paper": extracted_text
            }

        # ----------------------------------------------------
        # SAVE PAPER
        # ----------------------------------------------------

        connection = get_db_connection()

        cursor = connection.execute(
            """
            INSERT INTO papers
            (
                user_id,
                filename,
                extracted_text,
                sections,
                uploaded_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                user_id,
                original_filename,
                extracted_text,
                json.dumps(
                    section_with_content,
                    ensure_ascii=False
                ),
                datetime.now().isoformat()
            )
        )

        paper_id = cursor.lastrowid

        connection.commit()
        connection.close()

        # ----------------------------------------------------
        # CURRENT PAPER
        # ----------------------------------------------------

        session["paper_id"] = paper_id

        # ----------------------------------------------------
        # RETURN JSON
        # ----------------------------------------------------

        return jsonify({
            "success": True,
            "paper_id": paper_id,
            "topics": list(
                section_with_content.keys()
            )
        }), 200

    except Exception as error:

        print(
            "ERROR WHILE PROCESSING PDF:",
            repr(error)
        )

        return jsonify({
            "error": (
                "An error occurred while processing "
                "the PDF. Please try another PDF."
            )
        }), 500

    finally:

        # ----------------------------------------------------
        # PDF IS NO LONGER NEEDED AFTER EXTRACTION
        # ----------------------------------------------------

        try:

            if os.path.exists(filepath):

                os.remove(filepath)

        except OSError:

            pass


# ============================================================
# SUMMARY
# ============================================================

@app.route(
    "/summary",
    methods=["POST"]
)
def get_summary():

    if not login_required():

        return jsonify({
            "error": "Please login first."
        }), 401

    paper = get_current_paper()

    if not paper:

        return jsonify({
            "error": "No research paper is currently selected."
        }), 400

    data = request.get_json(
        silent=True
    ) or {}

    topic = data.get("topic")

    if not topic:

        return jsonify({
            "error": "No topic selected."
        }), 400

    sections = json.loads(
        paper["sections"]
    )

    topic_content = sections.get(
        topic
    )

    if not topic_content:

        return jsonify({
            "error": "Selected topic was not found."
        }), 404

    connection = get_db_connection()

    existing_summary = connection.execute(
        """
        SELECT summary
        FROM summaries
        WHERE paper_id = ?
        AND topic = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (
            paper["id"],
            topic
        )
    ).fetchone()

    connection.close()

    if existing_summary:

        return jsonify({
            "success": True,
            "summary": existing_summary["summary"]
        })

    try:

        summary = generate_detailed_summary(
            topic_content,
            get_llm()
        )

        connection = get_db_connection()

        connection.execute(
            """
            INSERT INTO summaries
            (
                paper_id,
                topic,
                summary,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                paper["id"],
                topic,
                summary,
                datetime.now().isoformat()
            )
        )

        connection.commit()
        connection.close()

        return jsonify({
            "success": True,
            "summary": summary
        })

    except Exception as error:

        print(
            "ERROR WHILE GENERATING SUMMARY:",
            repr(error)
        )

        return jsonify({
            "error": "Unable to generate summary."
        }), 500


# ============================================================
# CHAT
# ============================================================

@app.route(
    "/chat",
    methods=["POST"]
)
def chat():

    if not login_required():

        return jsonify({
            "error": "Please login first."
        }), 401

    paper = get_current_paper()

    if not paper:

        return jsonify({
            "error": "No research paper is currently selected."
        }), 400

    data = request.get_json(
        silent=True
    ) or {}

    user_message = data.get(
        "message",
        ""
    ).strip()

    if not user_message:

        return jsonify({
            "error": "Please enter a question."
        }), 400

    paper_id = paper["id"]

    try:

        # ----------------------------------------------------
        # CREATE LIGHTWEIGHT RETRIEVER
        # ----------------------------------------------------

        if paper_id not in vector_databases:

            print(
                f"Creating lightweight search index "
                f"for paper {paper_id}..."
            )

            vector_databases[paper_id] = (
                create_lightweight_vector_db(
                    paper["extracted_text"]
                )
            )

        vector_db = vector_databases[
            paper_id
        ]

        relevant_chunks = retrieve_relevant_chunks(
            vector_db,
            user_message,
            top_k=4
        )

        if not relevant_chunks:

            return jsonify({
                "success": True,
                "response": "I don't know."
            })

        context = "\n\n".join(
            relevant_chunks
        )

        # ----------------------------------------------------
        # ASK GROQ
        # ----------------------------------------------------

        prompt = f"""
You are a research paper assistant.

Answer the user's question ONLY using the
research paper context provided below.

Do not use outside knowledge.

If the answer is not present in the context,
reply exactly:

I don't know.

Research Paper Context:
{context}

User Question:
{user_message}

Answer clearly and concisely.
"""

        response = get_llm().invoke(
            prompt
        )

        if hasattr(response, "content"):

            answer = response.content.strip()

        else:

            answer = str(response).strip()

        return jsonify({
            "success": True,
            "response": answer
        })

    except Exception as error:

        print(
            "ERROR WHILE PROCESSING CHAT:",
            repr(error)
        )

        return jsonify({
            "error": "Unable to process your question."
        }), 500


# ============================================================
# ERROR HANDLER FOR LARGE FILES
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "error": "PDF is too large. Maximum size is 50 MB."
    }), 413


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    port = int(
        os.getenv(
            "PORT",
            "5000"
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )