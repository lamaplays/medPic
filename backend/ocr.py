import cv2
import pytesseract
import json



# Tesseract setup
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load SFDA brands
def load_sfda(json_path="sfda_cleaned.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return set(m["brand"].strip().lower() for m in data if m.get("brand"))

# Extract text from image
def process_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    h, w = thresh.shape
    resized = cv2.resize(thresh, (1000, int(1000 * h / w)))
    config = '--oem 3 --psm 6'
    return pytesseract.image_to_string(resized, config=config, lang='eng').strip()

