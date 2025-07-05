from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pii_processor import PIIProcessor
import os
from typing import List

router = APIRouter()
PROCESSED_FILES_DIR = "processed_files"

@router.post("/process-file")
async def process_file(
    file: UploadFile = File(...),
    mode: str = Form("mask")
):
    try:
        os.makedirs(PROCESSED_FILES_DIR, exist_ok=True)
        processor = PIIProcessor(mode=mode)
        
        # Process file
        content = await file.read()
        text_content = content.decode('utf-8', errors='replace')
        processed_content, entities = await processor.process_text(text_content)
        
        # Generate PDF
        output_path = processor.convert_to_pdf(
            processed_content,
            os.path.splitext(file.filename)[0]
        )
        
        # Get just filename for URL
        filename = os.path.basename(output_path)
        
        return {
            "message": "File processed successfully",
            "detected_entities": [e.dict() for e in entities],
            "pdf_download_url": f"/files/download/{filename}"  # Update path to include /files prefix
        }
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download/{filename}")
async def download_file(filename: str):
    """Download processed file"""
    try:
        file_path = os.path.join(PROCESSED_FILES_DIR, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
            
        return FileResponse(
            file_path,
            filename=filename,
            media_type="application/pdf"
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Error downloading file: {str(e)}")