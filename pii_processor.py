import pandas as pd
import re
from typing import Dict, List, Tuple, Union
from fastapi import UploadFile
import io
import email
from email.parser import BytesParser
from email import policy
from detect_pii import detect_pii, Entity, TextRequest
from fpdf import FPDF 
import os

class PIIProcessor:
    def __init__(self, mode: str = "mask"):
        self.mode = mode
        self.processed_dir = "processed_files"
        os.makedirs(self.processed_dir, exist_ok=True)
        self.mask_patterns = {
            'NAME': '[REDACTED-NAME]',
            'EMAIL': '[REDACTED-EMAIL]',
            'PHONE': '[REDACTED-PHONE]',
            'ADDRESS': '[REDACTED-ADDRESS]',
            'SSN': '[REDACTED-SSN]',
            'COMPANY': '[REDACTED-COMPANY]'
        }

    async def process_text(self, text: str) -> Tuple[str, List[Entity]]:
        """Process text and mask/remove PII"""
        try:
            # Create TextRequest
            request = TextRequest(text=text)
            
            # Get PII entities
            detected = await detect_pii(request)
            
            # Sort entities by position
            entities = sorted(detected.entities, key=lambda x: x.start if hasattr(x, 'start') else 0)
            
            # Modify text
            modified_text = text
            offset = 0
            
            for entity in entities:
                # Find entity position in text if start/end not provided
                if not hasattr(entity, 'start') or not hasattr(entity, 'end'):
                    start = modified_text.find(entity.text)
                    if start != -1:
                        entity.start = start
                        entity.end = start + len(entity.text)
                    else:
                        continue
                
                # Apply masking/removal with offset
                start = entity.start + offset
                end = entity.end + offset
                
                if self.mode == "mask":
                    replacement = self.mask_patterns.get(entity.label, f"[REDACTED-{entity.label}]")
                else:
                    replacement = ''
                    
                modified_text = modified_text[:start] + replacement + modified_text[end:]
                offset += len(replacement) - (end - start)
            
            return modified_text, detected.entities
            
        except Exception as e:
            print(f"Error in process_text: {str(e)}")
            raise

    async def process_email_file(self, file: UploadFile) -> Tuple[str, List[Entity]]:
        """Process email file"""
        content = await file.read()
        email_message = BytesParser(policy=policy.default).parsebytes(content)
        
        text_parts = []
        if email_message.is_multipart():
            for part in email_message.walk():
                if part.get_content_type() == "text/plain":
                    text_parts.append(part.get_payload())
        else:
            text_parts.append(email_message.get_payload())
        
        full_text = "\n".join(text_parts)
        return await self.process_text(full_text)

    async def process_csv_file(self, file: UploadFile) -> Tuple[pd.DataFrame, Dict[str, List[Entity]]]:
        """Process CSV file"""
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        processed_df = df.copy()
        all_entities = {}
        
        for column in df.columns:
            if df[column].dtype == 'object':
                for idx, value in df[column].items():
                    if pd.notna(value):
                        modified_text, entities = await self.process_text(str(value))
                        processed_df.at[idx, column] = modified_text
                        if entities:
                            all_entities[f"{column}_{idx}"] = entities
        
        return processed_df, all_entities

    def convert_to_pdf(self, content: Union[str, pd.DataFrame], filename: str) -> str:
        """Convert processed content to PDF with proper layout settings"""
        try:
            # Initialize PDF with proper page settings
            pdf = FPDF()
            pdf.add_page()
            
            # Set default font (no custom font needed)
            pdf.set_font("Arial", size=11)
            
            # Set margins
            pdf.set_left_margin(15)
            pdf.set_right_margin(15)
            pdf.set_auto_page_break(auto=True, margin=15)
            
            effective_page_width = pdf.w - 2 * pdf.l_margin
            
            if isinstance(content, str):
                # Process text content
                lines = content.split('\n')
                for line in lines:
                    try:
                        # Clean the text
                        clean_line = (line.encode('latin-1', 'replace')
                                    .decode('latin-1')
                                    .replace('\x00', '')
                                    .strip())
                        
                        if clean_line:  # Only process non-empty lines
                            # Calculate proper line height
                            line_height = pdf.font_size * 1.5
                            
                            # Add text with word wrap
                            pdf.multi_cell(
                                w=effective_page_width,
                                h=line_height,
                                txt=clean_line,
                                align='L'
                            )
                            
                            # Add small spacing between lines
                            pdf.ln(2)
                    except Exception as e:
                        print(f"Warning: Error writing line: {e}")
                        continue
            else:
                # Handle DataFrame
                try:
                    table_data = content.to_string()
                    pdf.multi_cell(
                        w=effective_page_width,
                        h=pdf.font_size * 1.5,
                        txt=table_data,
                        align='L'
                    )
                except Exception as e:
                    print(f"Warning: Error converting DataFrame: {e}")
                    pdf.multi_cell(
                        w=effective_page_width,
                        h=pdf.font_size * 1.5,
                        txt='[Error displaying content]'
                    )
            
            # Save PDF
            output_path = os.path.join(
                self.processed_dir,
                f"{filename}_processed.pdf"
            )
            pdf.output(output_path)
            
            return output_path
            
        except Exception as e:
            print(f"Error in convert_to_pdf: {str(e)}")
            raise