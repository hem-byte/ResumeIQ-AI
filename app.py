from flask import Flask, render_template, request, abort
from werkzeug.utils import secure_filename
import google.generativeai as genai
import PyPDF2
import os
import re

app = Flask(__name__)
if not os.path.exists("uploads"):
    os.makedirs("uploads")

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB upload limit

ALLOWED_EXTENSIONS = {"pdf"}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


genai.configure(
    api_key=os.environ.get("GEMINI_API_KEY")
)

model = genai.GenerativeModel("gemini-2.5-flash")


def extract_text(pdf_path):
    text = ""

    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text

    return text


def _clean_bullets(raw_text: str) -> str:
    """Normalize a block of text into one item per line with a leading dash.

    Removes numbering/bullets and empty lines, returns a newline-separated
    string where each item begins with `- ` so templates can render lists.
    """
    if not raw_text:
        return ""

    lines = []
    for line in raw_text.splitlines():
        item = line.strip()
        if not item:
            continue
        # remove common bullet characters and numeric prefixes
        item = re.sub(r'^[\-\*\u2022\•\s\d\)\.]+' , '', item).strip()
        if item:
            lines.append(f"- {item}")
    return "\n".join(lines)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    file = request.files["resume"]
    role = request.form["role"]

    if not file or file.filename == "":
        abort(400, "No file uploaded")

    if not allowed_file(file.filename):
        abort(400, "Only PDF files are allowed")

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    resume_text = extract_text(filepath)

    prompt = f"""
Analyze this resume for the role: {role}

Return EXACTLY in this format:

ATS_SCORE: <number>

STRENGTHS:
- point
- point
- point

MISSING_SKILLS:
- point
- point
- point

IMPROVEMENTS:
- point
- point
- point

FINAL_VERDICT:
One short paragraph

Resume:
{resume_text}
"""

    response = model.generate_content(prompt)

    result = response.text

    score = "0"

    score_match = re.search(r'ATS_SCORE:\s*([0-9]+(?:\.[0-9]+)?)', result, re.IGNORECASE)
    if score_match:
        # clamp/format score
        try:
            raw_score = float(score_match.group(1))
            score = str(int(max(0, min(100, raw_score))))
        except Exception:
            score = score_match.group(1)

    strengths = ""
    missing_skills = ""
    improvements = ""
    verdict = ""

    strengths_match = re.search(r'STRENGTHS:(.*?)MISSING_SKILLS:', result, re.DOTALL | re.IGNORECASE)
    missing_match = re.search(r'MISSING_SKILLS:(.*?)IMPROVEMENTS:', result, re.DOTALL | re.IGNORECASE)
    improvement_match = re.search(r'IMPROVEMENTS:(.*?)FINAL_VERDICT:', result, re.DOTALL | re.IGNORECASE)
    verdict_match = re.search(r'FINAL_VERDICT:(.*)', result, re.DOTALL | re.IGNORECASE)

    if strengths_match:
        strengths = _clean_bullets(strengths_match.group(1))

    if missing_match:
        missing_skills = _clean_bullets(missing_match.group(1))

    if improvement_match:
        improvements = _clean_bullets(improvement_match.group(1))

    if verdict_match:
        verdict = verdict_match.group(1).strip()

    # Fallbacks when model doesn't follow the exact format
    if not any([strengths, missing_skills, improvements, verdict]):
        # as a simple fallback, put entire result into verdict so user can see raw output
        verdict = result.strip()

    return render_template(
        "result.html",
        score=score,
        strengths=strengths,
        missing_skills=missing_skills,
        improvements=improvements,
        verdict=verdict
    )


if __name__ == "__main__":
    app.run(debug=True)