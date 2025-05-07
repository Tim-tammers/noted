from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response  # Changed from FileResponse
import cv2
import numpy as np
import io

app = FastAPI()
def smooth_handwriting(image_data: np.ndarray) -> np.ndarray:
    """Enhanced handwriting smoothing pipeline"""
    # Convert to grayscale if needed
    if len(image_data.shape) == 3:
        gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_data.copy()

    # 1. Noise reduction
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # 2. Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(denoised)

    # 3. Adaptive thresholding to isolate handwriting
    binary = cv2.adaptiveThreshold(
        contrast_enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        5
    )

    # 4. Stroke enhancement
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    smoothed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 5. Final processing
    _, final = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return final


async def smooth_image(file: UploadFile = File(...)):
    try:
        # Read and validate image
        contents = await file.read()
        if not contents:
            raise HTTPException(400, detail="Empty file received")
            
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            raise HTTPException(400, detail="Invalid image file")

        # Process image with enhanced smoothing
        processed_img = smooth_handwriting(img)
        
        # Encode to bytes
        success, encoded_img = cv2.imencode('.png', processed_img)
        if not success:
            raise HTTPException(500, detail="Failed to encode image")
        
        # Return as bytes with proper media type
        return Response(
            content=encoded_img.tobytes(),
            media_type="image/png"
        )
        
    except Exception as e:
        raise HTTPException(500, detail=str(e))