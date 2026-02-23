import os
import json
import time
import uuid
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    
    text = ""
    try:
        with open(file_path, 'rb') as file:
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        raise ValueError(f"Error reading PDF file: {e}")
    
    return text

def extract_json_from_response(response_text: str) -> dict:
    import re
    
    if hasattr(response_text, 'content'):
        response_text = response_text.content
    
    if not isinstance(response_text, str):
        response_text = str(response_text)
    
    json_pattern = r'```json\s*\n(.*?)\n```'
    match = re.search(json_pattern, response_text, re.DOTALL)
    
    if match:
        json_content = match.group(1)
    else:
        start_idx = response_text.find('{')
        if start_idx != -1:
            json_content = response_text[start_idx:response_text.rfind('}')+1]
        else:
            json_content = response_text
    
    json_content = re.sub(r',\s*]', ']', json_content)  # Remove trailing commas in arrays
    json_content = re.sub(r',\s*}', '}', json_content)  # Remove trailing commas in objects
    json_content = re.sub(r'//.*?\n', '', json_content)  # Remove comments
    json_content = re.sub(r'/\*.*?\*/', '', json_content, flags=re.DOTALL)  # Remove block comments
    
    try:
        return json.loads(json_content)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Raw content: {json_content}")
        json_content = re.sub(r'}\s*}', '}', json_content)
        return json.loads(json_content)
    except:
        return {}

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    cache=False
)

prompt = ChatPromptTemplate.from_template("""
You are an expert CV analyzer for automated MCQ generation. Extract structured candidate data **precisely**. Eliminate duplicates and ensure consistency.

Return a clean JSON with:

1) Basic information
- full_name (extract EXACT name as written in CV - no modifications)
- email (Extract the candidate's email address exactly as written in CV. The email may be embedded with special characters.
- phone (extract EXACT phone number as written in CV)

2) Experience summary
- experience_level (intern / junior / mid / senior)
- total_experience_duration (count months from formal work experience/internships only)

3) Technical profile
- primary_domains (max 5 main domains)
- core_skills (most demonstrated skills)
- secondary_skills (less demonstrated skills)
- skill_levels (beginner / intermediate / advanced for each skill)

4) Technology classification
Automatically group all technical skills into logical categories based on their function and context. Create categories dynamically based on technologies found in the CV. Use these standard categories and place each technology in the most appropriate one:
- Programming languages (Python, Java, JavaScript, TypeScript, C, C++, PHP, SQL)
- Web frameworks (React, Angular, Django, Express, Next.js, Nuxt.js, Spring Boot, Symfony, FastAPI)
- Databases (MySQL, PostgreSQL, MongoDB, etc.)
- AI/ML libraries (TensorFlow, OpenCV, NumPy, Pandas, Streamlit, etc.)
- Cloud platforms (AWS, Azure, GCP, etc.)
- Development tools (VS Code, Git, GitHub, Postman, Figma, etc.)
- Deployment platforms (Vercel, Netlify, Heroku, etc.)
- Collaboration tools (Slack, Notion, ClickUp, etc.)
- Desktop/GUI development (Java Swing, JavaFX, Electron, etc.)
- Mobile development (React Native, Flutter, Swift, Kotlin, etc.)
- Other specialized categories as needed based on specific technologies

CRITICAL: 
- No technology should appear in multiple categories
- No empty categories should be created
- Each technology must be placed in exactly one appropriate category based on its primary function

5) Project & experience analysis
- strongest_technologies (technologies most used across projects and internships)
- main_project_types (e.g., web application, mobile app, AI system, API, etc.)
- responsibility_level (academic / personal / professional / internship)

6) Education
- highest_degree (extract EXACT degree name as written in CV)
- current_field_of_study (extract EXACT field of study as written in CV. If not explicitly mentioned, return null)

7) Additional information
- projects_list (ALL projects with name and brief description - include academic projects, internships, personal projects, professional work)
- positions_sought (ONLY positions explicitly mentioned in CV - return empty array if none found)
- internships (ALL internships with company, position, duration - extract duration as "X months" or "X years" format for clarity)

Rules:
- Extract ONLY what is explicitly stated in CV
- Ensure consistency between technical profile and technology classification
- For duration: calculate precisely from dates provided in CV
- Return strictly valid JSON

CV:
{cv_text}
""")

chain = prompt | llm

unique_id = str(uuid.uuid4())[:8]
timestamp = int(time.time())

cv_text = extract_text_from_pdf("wala_oueslati.pdf")

cv_text_with_context = f"""
CV_PROCESSING_ID:{unique_id}_TIME:{timestamp}
{cv_text}
"""

result = chain.invoke({"cv_text": cv_text_with_context})

parsed_data = extract_json_from_response(result)

if parsed_data:
    print(json.dumps(parsed_data, indent=2, ensure_ascii=False))
    
    with open("cv_content.json", "w", encoding="utf-8") as json_file:
        json.dump(parsed_data, json_file, indent=2, ensure_ascii=False)
    print("CV content saved to 'cv_content.json'")
    
else:
    print("Failed to extract valid JSON from the response")

