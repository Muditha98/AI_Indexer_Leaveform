from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional, Union
import json
import asyncio
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
import os
from dotenv import load_dotenv
import base64
from PyPDF2 import PdfReader, PdfWriter
import io
from concurrent.futures import ThreadPoolExecutor
import logging
from datetime import datetime
import re
from dateutil import parser
import pytz

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('document_processor.log')
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

# Validate environment variables
required_env_vars = ["AZURE_ENDPOINT", "AZURE_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

app = FastAPI()

# Updated CORS configuration allowing all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Azure Configuration
try:
    document_analysis_client = DocumentAnalysisClient(
        endpoint=os.getenv("AZURE_ENDPOINT"),
        credential=AzureKeyCredential(os.getenv("AZURE_KEY"))
    )
    logger.info("Successfully initialized Document Analysis Client")
except Exception as e:
    logger.error(f"Failed to initialize Document Analysis Client: {str(e)}")
    raise

# Initialize ThreadPoolExecutor for CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=4)

# Field mappings
FIELD_MAPPINGS = {
    "I.D. No.": "ID Number",
    "Employee Name": "Employee Name",
    "Date Filed": "Date Filed",
    "Reason For Leave:": "Reason"
}

KEYS_OF_INTEREST = list(FIELD_MAPPINGS.keys())

class DateParsingError(Exception):
    """Custom exception for date parsing errors"""
    pass

def clean_date_string(date_str: str) -> str:
    """Clean and normalize date string before parsing."""
    if not date_str:
        raise DateParsingError("Empty date string")
    
    date_str = str(date_str).strip()
    date_str = re.sub(r'\s+', ' ', date_str)
    
    month_replacements = {
        'january': '01', 'jan': '01',
        'february': '02', 'feb': '02',
        'march': '03', 'mar': '03',
        'april': '04', 'apr': '04',
        'may': '05',
        'june': '06', 'jun': '06',
        'july': '07', 'jul': '07',
        'august': '08', 'aug': '08',
        'september': '09', 'sep': '09',
        'october': '10', 'oct': '10',
        'november': '11', 'nov': '11',
        'december': '12', 'dec': '12'
    }
    
    lower_date = date_str.lower()
    for month_str, month_num in month_replacements.items():
        if month_str in lower_date:
            lower_date = lower_date.replace(month_str, month_num)
    
    cleaned = re.sub(r'[^\w\s/-]', '', lower_date)
    return cleaned

def standardize_date_format(date_str: str, output_format: str = "%d/%m/%y") -> str:
    """Convert various date formats to specified output format."""
    try:
        cleaned_date = clean_date_string(date_str)
        
        date_formats = [
            "%d/%m/%y", "%d/%m/%Y",
            "%m/%d/%y", "%m/%d/%Y",
            "%Y-%m-%d", "%d-%m-%Y",
            "%Y/%m/%d", "%d/%m/%Y",
            "%d %m %Y", "%m %d %Y",
            "%Y %m %d", "%d %b %Y",
            "%Y-%m-%dT%H:%M:%S",
            "%Y%m%d"
        ]
        
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(cleaned_date, fmt)
                return parsed_date.strftime(output_format)
            except ValueError:
                continue
        
        try:
            parsed_date = parser.parse(cleaned_date)
            return parsed_date.strftime(output_format)
        except (ValueError, parser.ParserError):
            raise DateParsingError(f"Unable to parse date: {date_str}")
            
    except Exception as e:
        logger.error(f"Date standardization error for '{date_str}': {str(e)}")
        raise DateParsingError(f"Date parsing failed: {str(e)}")

async def extract_first_page(file_content: bytes) -> Optional[str]:
    """Extract first page from PDF and return as a thumbnail image in base64 encoded string using PyMuPDF."""
    def _extract():
        try:
            logger.info(f"Starting PDF first page extraction with content size: {len(file_content)} bytes")
            
            # Import fitz (PyMuPDF) inside the function to handle any import issues gracefully
            import fitz
            
            # Load PDF from bytes
            pdf = fitz.open(stream=file_content, filetype="pdf")
            
            if len(pdf) == 0:
                logger.error("PDF has no pages")
                return None
            
            # Get first page
            page = pdf[0]
            logger.info(f"Processing page {page.number + 1}, size: {page.rect}")
            
            # Render page to an image with higher resolution for better quality
            # Adjust the matrix values for higher resolution
            zoom_factor = 2.0  # Adjust as needed for quality vs file size
            mat = fitz.Matrix(zoom_factor, zoom_factor)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            logger.info(f"Generated pixmap with dimensions: {pix.width}x{pix.height}")
            
            # Convert to PNG data
            img_bytes = pix.tobytes("png")
            
            # Convert to base64
            thumbnail = base64.b64encode(img_bytes).decode('utf-8')
            logger.info(f"Generated thumbnail of size: {len(thumbnail)} bytes")
            
            return thumbnail
                
        except ImportError as e:
            logger.error(f"PyMuPDF import error: {str(e)}. Please install with 'pip install pymupdf'")
            return None
        except Exception as e:
            logger.error(f"PDF extraction error: {str(e)}", exc_info=True)
            # Log additional details for troubleshooting
            logger.error(f"PDF content size: {len(file_content)} bytes")
            if len(file_content) > 0:
                # Log the first few bytes of the file to help diagnose issues
                try:
                    logger.error(f"PDF content first 100 bytes (hex): {file_content[:100].hex()}")
                except:
                    pass
            return None

    return await asyncio.get_event_loop().run_in_executor(executor, _extract)

async def process_form_recognizer(file_content: bytes, filename: str) -> Dict:
    """Process document with Form Recognizer and map field names."""
    try:
        logger.info(f"Starting Form Recognizer analysis for file: {filename}")
        
        poller = document_analysis_client.begin_analyze_document(
            "prebuilt-document",
            document=file_content
        )
        result = poller.result()
        
        extracted_data = {}
        for kv_pair in result.key_value_pairs:
            if kv_pair.key and kv_pair.value:
                key = kv_pair.key.content.strip()
                if key in KEYS_OF_INTEREST:
                    mapped_key = FIELD_MAPPINGS[key]
                    value = kv_pair.value.content.strip()
                    
                    if mapped_key == "Date Filed":
                        try:
                            value = standardize_date_format(value)
                            logger.info(f"Successfully standardized date: {value}")
                        except DateParsingError as e:
                            logger.warning(f"Date parsing failed for {value}: {str(e)}")
                    
                    extracted_data[mapped_key] = {
                        'value': value,
                        'confidence': getattr(kv_pair, 'confidence', None)
                    }
        
        return extracted_data

    except Exception as e:
        logger.error(f"Form recognizer error for {filename}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process-document")
async def process_document(
    file: UploadFile = File(...),
    metadata: str = Form(...),
    token: str = Form(...)
):
    """Process a single document with optimized concurrent processing."""
    try:
        logger.info(f"Processing single document: {file.filename}")
        
        file_content = await file.read()
        metadata_dict = json.loads(metadata)
        
        # Process form recognition and first page extraction concurrently
        extracted_data_task = process_form_recognizer(file_content, file.filename)
        thumbnail_task = extract_first_page(file_content)
        
        # Wait for both tasks to complete
        extracted_data, thumbnail = await asyncio.gather(
            extracted_data_task,
            thumbnail_task
        )
        
        if thumbnail is None:
            logger.warning(f"Thumbnail generation failed for {file.filename}")
        
        response_data = {
            "filename": file.filename,
            "extracted_data": extracted_data,
            "thumbnail": thumbnail,
            "metadata": metadata_dict
        }
        
        return JSONResponse(
            content=response_data,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS, GET",
                "Access-Control-Allow-Headers": "*"
            }
        )
    
    except Exception as e:
        logger.error(f"Error processing document {file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process-documents-batch")
async def process_documents_batch(
    files: List[UploadFile] = File(...),
    metadata: str = Form(...),
    token: str = Form(...)
):
    """Process multiple documents concurrently."""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum batch size is 10 documents")
    
    try:
        logger.info(f"Processing batch of {len(files)} documents")
        metadata_dict = json.loads(metadata)
        
        async def process_single_document(file: UploadFile):
            try:
                file_content = await file.read()
                extracted_data_task = process_form_recognizer(file_content, file.filename)
                thumbnail_task = extract_first_page(file_content)
                
                extracted_data, thumbnail = await asyncio.gather(
                    extracted_data_task,
                    thumbnail_task
                )
                
                return {
                    "filename": file.filename,
                    "extracted_data": extracted_data,
                    "thumbnail": thumbnail,
                    "metadata": metadata_dict,
                    "status": "success"
                }
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {str(e)}")
                return {
                    "filename": file.filename,
                    "status": "error",
                    "error": str(e)
                }
        
        tasks = [process_single_document(file) for file in files]
        results = await asyncio.gather(*tasks)
        
        response_data = {
            "batch_size": len(files),
            "results": results,
            "successful": len([r for r in results if r.get("status") == "success"]),
            "failed": len([r for r in results if r.get("status") == "error"])
        }
        
        return JSONResponse(
            content=response_data,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS, GET",
                "Access-Control-Allow-Headers": "*"
            }
        )
        
    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.options("/api/process-document")
async def options_process_document():
    """Handle preflight requests for the process-document endpoint."""
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS, GET",
            "Access-Control-Allow-Headers": "*"
        }
    )

@app.options("/api/process-documents-batch")
async def options_process_documents_batch():
    """Handle preflight requests for the batch processing endpoint."""
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS, GET",
            "Access-Control-Allow-Headers": "*"
        }
    )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Verify Azure Form Recognizer client
        credential = document_analysis_client.credential
        
        # Test date parsing functionality
        test_dates = ["2024-01-01", "01/01/24", "January 1, 2024"]
        date_parsing_status = all(
            standardize_date_format(date) for date in test_dates
        )
        
        return {
            "status": "healthy",
            "azure_client": "configured",
            "environment": "all required variables set",
            "date_parsing": "operational" if date_parsing_status else "warning"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Service unhealthy: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI application")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
    