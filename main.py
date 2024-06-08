from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import PyPDF2
import google.generativeai as genai
import tempfile
import os
import pytesseract
from PIL import Image
from google.generativeai.types import HarmCategory, HarmBlockThreshold

app = FastAPI()

# Configure Google Generative AI
GOOGLE_API_KEY = 'AIzaSyCAzjRDfy9rbkP4v8CWCi9_vWaypLPY15c'
genai.configure(api_key=GOOGLE_API_KEY)

# Specify the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    return text

def handle_pdf(pdf_path, subject, model):
    extracted_text = extract_text_from_pdf(pdf_path)

    response = model.generate_content(
        [f'I have extracted text from a pdf, which is my {subject} assignment. Please answer these questions:{extracted_text}'],
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
        }
    )
    
    return response.text

def extract_text_from_img(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

def handle_image(image_path, subject, model):
    extracted_text = extract_text_from_img(image_path)

    response = model.generate_content(
        [f'I have extracted text from an image, which is my {subject} assignment. Please answer these questions:{extracted_text}'],
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
        }
    )
    
    return response.text

@app.post("/process-file/")
async def process_file(file: UploadFile = File(...), subject: str = Form(...)):
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Save the uploaded file to the temporary directory
        file_path = os.path.join(tmpdirname, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Initialize the Generative AI model
        model = genai.GenerativeModel(model_name="gemini-pro")

        try:
            # Determine file type and process accordingly
            if file.filename.lower().endswith(".pdf"):
                response = handle_pdf(file_path, subject, model)
            elif file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
                response = handle_image(file_path, subject, model)
            else:
                raise ValueError("Unsupported file type. Please provide a PDF or image file.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    return JSONResponse(content={"response": response})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
