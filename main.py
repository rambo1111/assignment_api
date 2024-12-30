from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import PyPDF2
import google.generativeai as genai
import tempfile
import os
import easyocr
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from all origins
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Configure Google Generative AI
GOOGLE_API_KEY = 'AIzaSyD1GavBmslusEMDZdynxZXM6dUEtia7FwM'
genai.configure(api_key=GOOGLE_API_KEY)

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise
    return text

def handle_pdf(pdf_path, subject, model):
    try:
        extracted_text = extract_text_from_pdf(pdf_path)

        response = model.generate_content(
            [f'I have extracted text from a pdf, which is my {subject} assignment. Please answer these questions: {extracted_text}'],
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
            }
        )
    except Exception as e:
        logger.error(f"Error handling PDF: {e}")
        raise
    return response.text

def extract_text_from_img(image_path):
    text = ""
    try:
        reader = easyocr.Reader(['en'])
        results = reader.readtext(image_path)
        text = " ".join([text for _, text, _ in results])
    except Exception as e:
        logger.error(f"Error extracting text from image: {e}")
        raise
    return text

def handle_image(image_path, subject, model):
    try:
        extracted_text = extract_text_from_img(image_path)

        response = model.generate_content(
            [f'I have extracted text from an image, which is my {subject} assignment. Please answer these questions: {extracted_text}'],
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
            }
        )
    except Exception as e:
        logger.error(f"Error handling image: {e}")
        raise
    return response.text

@app.post("/process-file/")
async def process_file(file: UploadFile = File(...), subject: str = Form(...)):
    logger.info("Received file for processing")
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Save the uploaded file to the temporary directory
        file_path = os.path.join(tmpdirname, file.filename)
        try:
            with open(file_path, "wb") as f:
                f.write(await file.read())

            # Initialize the Generative AI model
            model = genai.GenerativeModel(model_name="gemini-pro")

            # Determine file type and process accordingly
            if file.filename.lower().endswith(".pdf"):
                response = handle_pdf(file_path, subject, model)
            elif file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
                response = handle_image(file_path, subject, model)
            else:
                raise ValueError("Unsupported file type. Please provide a PDF or image file.")
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    return JSONResponse(content={"response": response})

@app.get("/healthz")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
