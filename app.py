from flask import Flask, render_template, request
import google.generativeai as genai
import PyPDF2
import os
import re

app = Flask(__name__)
if not os.path.exists("uploads"):
    os.makedirs("uploads")

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


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


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    file = request.files["resume"]
    role = request.form["role"]

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

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

    score_match = re.search(
        r'ATS_SCORE:\s*(\d+)',
        result
    )

    if score_match:
        score = score_match.group(1)

    strengths = ""
    missing_skills = ""
    improvements = ""
    verdict = ""

    strengths_match = re.search(
        r'STRENGTHS:(.*?)MISSING_SKILLS:',
        result,
        re.DOTALL
    )

    missing_match = re.search(
        r'MISSING_SKILLS:(.*?)IMPROVEMENTS:',
        result,
        re.DOTALL
    )

    improvement_match = re.search(
        r'IMPROVEMENTS:(.*?)FINAL_VERDICT:',
        result,
        re.DOTALL
    )

    verdict_match = re.search(
        r'FINAL_VERDICT:(.*)',
        result,
        re.DOTALL
    )

    if strengths_match:
        strengths = strengths_match.group(1)

    if missing_match:
        missing_skills = missing_match.group(1)

    if improvement_match:
        improvements = improvement_match.group(1)

    if verdict_match:
        verdict = verdict_match.group(1)

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