from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import PyPDF2
import google.generativeai as genai
import tempfile
import os
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Configure Google Generative AI
GOOGLE_API_KEY = 'AIzaSyCAzjRDfy9rbkP4v8CWCi9_vWaypLPY15c'
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize FastAPI app
app = FastAPI()

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Endpoint to handle PDF upload and subject input
@app.post("/process_pdf/")
async def process_pdf(file: UploadFile = File(...), subject: str = Form(...)):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(await file.read())
        temp_pdf_path = temp_pdf.name

    # Extract text from the PDF
    extracted_text = extract_text_from_pdf(temp_pdf_path)
    
    # Remove the temporary file
    os.remove(temp_pdf_path)
    
    # Generate content using Google Generative AI
    model = genai.GenerativeModel(model_name="gemini-pro")
    response = model.generate_content(
        [f'I have extracted text from a pdf, which is my {subject} assignment. Please answer these questions, define the following: {extracted_text}'],
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
        }
    )
        
    return JSONResponse(content=response.text)

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
