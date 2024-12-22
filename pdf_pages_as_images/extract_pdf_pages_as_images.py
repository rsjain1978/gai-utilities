import os
from typing import List, Dict, Optional
import json
from pathlib import Path
import asyncio
from pdf2image import convert_from_path
import base64
from io import BytesIO
from PIL import Image
from openai import AsyncOpenAI
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('execution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Extractor:
        
    def convert_pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """Convert PDF pages to images with error handling and save to folder."""
        try:
            output_dir = Path('extracted_pages')
            
            # Check if directory exists and contains images
            if output_dir.exists():

                logger.info("Existing images found. Loading images...")

                existing_images = sorted(output_dir.glob('page_*.png'))
                if existing_images:
                    logger.info(f"Found {len(existing_images)} existing images in {output_dir}")
                    # Load existing images
                    images = [Image.open(img_path) for img_path in existing_images]
                    return images
            
            # If no existing images found, create directory and convert PDF
            output_dir.mkdir(exist_ok=True)
            
            # Convert PDF to images
            logger.info("No existing images found. Converting PDF to images...")
            images = convert_from_path(
                pdf_path,
                dpi=300,  # Higher DPI for better quality
                fmt="PNG"
            )
            logger.info(f"Successfully converted PDF to {len(images)} images")
            
            # Save each image
            for i, image in enumerate(images, 1):
                image_path = output_dir / f"page_{i:03d}.png"
                image.save(image_path, "PNG", optimize=True)
                logger.info(f"Saved page {i} to {image_path}")
                
            return images
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
            raise

async def main():
    
    # Replace with your PDF file path
    pdf_path = "Momentum_Picks.pdf"
    
    extractor = Extractor()
    extractor.convert_pdf_to_images(pdf_path)


if __name__ == "__main__":
    asyncio.run(main()) 