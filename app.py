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

from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from src.load_and_extract_text import (
    extract_text_from_pdf,
    extract_pdf_sections
)

from src.detect_and_split_sections import (
    refine_sections,
    split_sections_with_content
)

from src.get_summary import generate_detailed_summary
from src.create_vector_db import create_vector_db
from src.RAG_retrival_chain import get_qa_chain


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

# Folder where uploaded PDFs are temporarily stored
app.config["UPLOAD_FOLDER"] = os.getenv(
    "UPLOAD_FOLDER",
    "uploads"
)

# Maximum upload size: 50 MB
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


# ============================================================
# FLASK SECRET KEY
# ============================================================

flask_secret_key = os.getenv("FLASK_SECRET_KEY")

if not flask_secret_key:
    print(
        "WARNING: FLASK_SECRET_KEY is not configured. "
        "Using development fallback."
    )

    flask_secret_key = (
        "research-paper-analyzer-development-key"
    )

app.secret_key = flask_secret_key


# ============================================================
# SESSION COOKIE CONFIGURATION
# ============================================================

cookie_secure = (
    os.getenv("COOKIE_SECURE", "false").lower() == "true"
)

app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = cookie_secure


# ============================================================
# CREATE UPLOAD FOLDER
# ============================================================

os.makedirs(
    app.config["UPLOAD_FOLDER"],
    exist_ok=True
)


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DATABASE = os.getenv(
    "DATABASE_PATH",
    "research_paper_analyzer.db"
)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db_connection():
    """
    Create and return a SQLite database connection.
    """

    connection = sqlite3.connect(DATABASE)

    # Allows rows to be accessed like dictionaries
    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def initialize_database():
    """
    Create all required database tables if they do not exist.
    """

    connection = get_db_connection()
    cursor = connection.cursor()

    # --------------------------------------------------------
    # USERS TABLE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # PAPERS TABLE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # SUMMARIES TABLE
    # --------------------------------------------------------

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


# Initialize database when application starts
initialize_database()


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

groq_api_key = os.getenv("GROQ_API_KEY")
llm_model = os.getenv("LLM_MODEL")
embedding_model = os.getenv("EMBEDDING_MODEL")


# ============================================================
# CHECK REQUIRED ENVIRONMENT VARIABLES
# ============================================================

if not groq_api_key:
    print("WARNING: GROQ_API_KEY is not configured.")

if not llm_model:
    print("WARNING: LLM_MODEL is not configured.")

if not embedding_model:
    print("WARNING: EMBEDDING_MODEL is not configured.")


# ============================================================
# INITIALIZE LLM AND EMBEDDINGS
# ============================================================

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name=llm_model
)

embedder = HuggingFaceEmbeddings(
    model_name=embedding_model
)


# ============================================================
# IN-MEMORY VECTOR DATABASE CACHE
# ============================================================

vector_databases = {}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_current_user():
    """
    Return the currently logged-in user from the database.

    If the session contains a user_id that no longer exists
    in the database, clear the stale session and return None.

    Returns:
        sqlite3.Row or None
    """

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

    # --------------------------------------------------------
    # IMPORTANT:
    # Handle stale sessions after Render restart/database reset.
    # --------------------------------------------------------

    if user is None:
        session.clear()
        return None

    return user


def login_required():
    """
    Check whether a valid logged-in user exists.

    This checks both:
    1. Whether user_id exists in the session.
    2. Whether that user actually exists in the database.

    Returns:
        True or False
    """

    user = get_current_user()

    return user is not None


def get_current_paper():
    """
    Get the currently selected paper for the logged-in user.
    """

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
        (paper_id, user_id)
    ).fetchone()

    connection.close()

    return paper


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    user = get_current_user()

    if user is None:
        return redirect(url_for("login"))

    return render_template(
        "index.html",
        username=user["username"]
    )


# ============================================================
# REGISTER
# ============================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    # If already logged in, go to dashboard
    if login_required():
        return redirect(url_for("index"))

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

        # --------------------------------------------
        # Validation
        # --------------------------------------------

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

        # --------------------------------------------
        # Hash password
        # --------------------------------------------

        password_hash = generate_password_hash(password)

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

        return redirect(url_for("login"))

    return render_template("register.html")


# ============================================================
# LOGIN
# ============================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if login_required():
        return redirect(url_for("index"))

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

            # Clear old session
            session.clear()

            # Store logged-in user
            session["user_id"] = user["id"]
            session["username"] = user["username"]

            return redirect(url_for("index"))

        return render_template(
            "login.html",
            error="Invalid username or password."
        )

    return render_template("login.html")


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("login"))


# ============================================================
# PREVIOUS WORK / HISTORY
# ============================================================

@app.route("/history")
def history():

    if not login_required():
        return redirect(url_for("login"))

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
# VIEW PREVIOUS PAPER
# ============================================================

@app.route("/history/<int:paper_id>")
def view_history(paper_id):

    if not login_required():
        return redirect(url_for("login"))

    user_id = session["user_id"]

    connection = get_db_connection()

    paper = connection.execute(
        """
        SELECT *
        FROM papers
        WHERE id = ?
        AND user_id = ?
        """,
        (paper_id, user_id)
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

@app.route("/upload", methods=["POST"])
def upload_pdf():

    if not login_required():
        return jsonify({
            "error": "Please login first."
        }), 401

    user_id = session["user_id"]

    file = request.files.get("file")

    if not file:
        return jsonify({
            "error": "No file uploaded."
        }), 400

    if not file.filename:
        return jsonify({
            "error": "No file selected."
        }), 400

    # --------------------------------------------------------
    # Check file extension
    # --------------------------------------------------------

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({
            "error": "Only PDF files are allowed."
        }), 400

    # --------------------------------------------------------
    # Create unique filename
    # --------------------------------------------------------

    original_filename = file.filename

    unique_filename = (
        str(uuid.uuid4()) +
        ".pdf"
    )

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        unique_filename
    )

    file.save(filepath)

    try:

        # ----------------------------------------------------
        # Extract complete text
        # ----------------------------------------------------

        extracted_text = extract_text_from_pdf(
            filepath
        )

        if not extracted_text or not extracted_text.strip():

            return jsonify({
                "error": "Could not extract text from this PDF."
            }), 400

        # ----------------------------------------------------
        # Detect sections
        # ----------------------------------------------------

        extracted_sections = extract_pdf_sections(
            full_text=extracted_text
        )

        # ----------------------------------------------------
        # Refine sections using LLM
        # ----------------------------------------------------

        refined_sections = refine_sections(
            extracted_sections,
            llm
        )

        # ----------------------------------------------------
        # Split paper into sections
        # ----------------------------------------------------

        section_with_content = split_sections_with_content(
            extracted_text,
            refined_sections
        )

        # ----------------------------------------------------
        # Save paper information
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
        # Set current paper
        # ----------------------------------------------------

        session["paper_id"] = paper_id

        # ----------------------------------------------------
        # Return topics
        # ----------------------------------------------------

        return jsonify({
            "success": True,
            "paper_id": paper_id,
            "topics": list(
                section_with_content.keys()
            )
        })

    except Exception as error:

        print(
            "Error while processing PDF:",
            error
        )

        # Remove uploaded file if processing failed
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except OSError:
            pass

        return jsonify({
            "error": "An error occurred while processing the PDF."
        }), 500


# ============================================================
# GENERATE SUMMARY
# ============================================================

@app.route("/summary", methods=["POST"])
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

    topic_content = sections.get(topic)

    if not topic_content:
        return jsonify({
            "error": "Selected topic was not found."
        }), 404

    # --------------------------------------------------------
    # Check whether summary already exists
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Generate new summary
    # --------------------------------------------------------

    summary = generate_detailed_summary(
        topic_content,
        llm
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

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


# ============================================================
# CHAT WITH CURRENT PAPER
# ============================================================

@app.route("/chat", methods=["POST"])
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
        # Create vector database if not already available
        # ----------------------------------------------------

        if paper_id not in vector_databases:

            print(
                f"Creating vector database for paper {paper_id}..."
            )

            vector_db = create_vector_db(
                text=paper["extracted_text"],
                embedder=embedder
            )

            vector_databases[paper_id] = vector_db

        else:

            vector_db = vector_databases[
                paper_id
            ]

        # ----------------------------------------------------
        # Create QA chain
        # ----------------------------------------------------

        chain = get_qa_chain(
            vectordb=vector_db,
            llm=llm
        )

        # ----------------------------------------------------
        # Ask question
        # ----------------------------------------------------

        result = chain.invoke(
            {
                "query": user_message
            }
        )

        ai_response = result.get(
            "result",
            "I don't know."
        )

        return jsonify({
            "success": True,
            "response": ai_response
        })

    except Exception as error:

        print(
            "Error while processing chat:",
            error
        )

        return jsonify({
            "error": "Unable to process your question."
        }), 500


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
        debug=False,
        host="0.0.0.0",
        port=port
    )