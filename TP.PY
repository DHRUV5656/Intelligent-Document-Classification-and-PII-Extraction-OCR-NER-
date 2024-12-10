import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import io
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import re
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import tensorflow as tf
import sys
import cv2
import matplotlib.pyplot as plt
import random
import string

tf.config.optimizer.set_jit(True)  # Enable XLA

# Environment settings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Constants
IMG_WIDTH = 150
IMG_HEIGHT = 150
MAX_WIDTH = 800
MAX_HEIGHT = 600
DPI = 300

# Labels for classification
labels = {
    0: "Aadhaar",
    1: "Canada ID",
    2: "Czech ID",
    3: "Denmark PID",
    4: "Driving Licence",
    5: "Finland ID",
    6: "Israel NID",
    7: "PAN",
    8: "Passport",
    9: "Poland ID",
    10: "Singapore ID",
    11: "South Africa ID",
    12: "Spain SSN",
    13: "UK DL",
    14: "US Passport",
    15: "Utility",
    16: "Voter ID"
}

def resource_path(relative_path):
    """ Get the absolute path to the resource, works for dev and for PyInstaller bundle """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

model_path = resource_path(r"C:\Users\HP\Desktop\quick\foreign_indian_classificiation_model.keras")

# Load the model
cnn_model = load_model(model_path)

def preprocess_image_classification(img, img_width=IMG_WIDTH, img_height=IMG_HEIGHT):
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def perform_ocr(img):
    custom_config = '--psm 11 --oem 3'
    return pytesseract.image_to_string(img, lang='eng+hin+mar', config=custom_config)

def extract_info(text, doc_type):
    patterns = {
        'Aadhaar': r'''(?<!\d)((?:[०-९]{4}[-\s]?[०-९]{4}[-\s]?[०-९]{4})|(?:[௦-௯]{4}[-\s]?[௦-௯]{4}[-\s]?[௦-௯]{4})|(?:[౦-౯]{4}[-\s]?[౦-౯]{4}[-\s]?[౦-౯]{4})|(?:[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}))(?!\d)''',
        'Canada ID': r'\b\d{9}\b',
        'Czech ID': r'\b[A-Z]{1}[0-9]{6}[A-Z]{1}\b',
        'Denmark PID': r'\b\d{10}\b',
        'Driving Licence': r'([A-Z]{2}\d{2}[- ]?\d{4}[- ]?[A-Z]{2}\b)',
        'Finland ID': r'\b\d{6}-\d{4}\b',
        'Israel NID': r'\b\d{9}\b',
        'PAN': r'([A-Z]{5}[0-9]{4}[A-Z]{1})',
        'Passport': r'([A-Z]{1}[0-9]{7})',
        'Poland ID': r'\b[A-Z]{2}[0-9]{7}\b',
        'Singapore ID': r'\b[S|T|F|G][0-9]{7}[A-Z]\b',
        'South Africa ID': r'\b\d{13}\b',
        'Spain SSN': r'\b\d{8}-[A-Z]\b',
        'UK DL': r'\b[A-Z]{2}\d{2}[- ]?\d{4}[- ]?[A-Z]{2}\b',
        'US Passport': r'\b[0-9]{9}\b',
        'Utility': r'\b(?:Aadhaar|Driving Licence|PAN|Passport|Canada ID|Czech ID|Denmark PID|Finland ID|Israel NID|Poland ID|Singapore ID|South Africa ID|Spain SSN|UK DL|US Passport|Voter ID)\b',
        'Voter ID': r'\b[A-Za-z]{3}[0-9]{7}\b'
    }

    extracted_info = {key: match.group(0) for key, pattern in patterns.items() if (match := re.search(pattern, text))}
    for key, value in extracted_info.items():
        print(f"{key}: {value}")
    
    return extracted_info

def reclassify_utility(text):
    keywords = {
        'Aadhaar': ['Aadhaar', 'UIDAI'],
        'Driving Licence': ['Driving Licence', 'DL No'],
        'PAN': ['Permanent Account Number', 'Income Tax Department'],
        'Passport': ['Passport', 'Republic of India'],
        'US Passport': ['Passport', 'United States of America'],
        'Voter ID': ['ELECTION','COMMISSION','ELECTION COMMISSION OF INDIA','IDENTITY CARD'],
        'Canada ID': ['Canada ID'],
        'Czech ID': ['Czech ID'],
        'Denmark PID': ['Denmark PID'],
        'Finland ID': ['Finland ID'],
        'Israel NID': ['Israel NID'],
        'Poland ID': ['Poland ID'],
        'Singapore ID': ['Singapore ID'],
        'South Africa ID': ['South Africa ID'],
        'Spain SSN': ['Spain SSN']
    }

    for doc_type, words in keywords.items():
        if any(re.search(word, text, re.IGNORECASE) for word in words):
            print(f"Keyword found for document type '{doc_type}'")
            return doc_type
    return 'Utility'

def process_pdf_and_extract_images(pdf_path):
    pdf_reader = fitz.open(pdf_path)
    images = []

    for page_num in range(len(pdf_reader)):
        page = pdf_reader.load_page(page_num)
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            base_image = pdf_reader.extract_image(xref)
            img = Image.open(io.BytesIO(base_image["image"])).convert("RGBA")
            images.append(img)

    pdf_reader.close()
    return images

def resize_image(img, max_width=MAX_WIDTH, max_height=MAX_HEIGHT):
    width, height = img.size
    if width > max_width or height > max_height:
        scaling_factor = min(max_width / width, max_height / height)
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        img = img.resize(new_size, Image.LANCZOS)
        img.info['dpi'] = (DPI, DPI)  
    return img

def process_single_image(img):
    img_array = preprocess_image_classification(img)
    predictions = cnn_model.predict(img_array)
    predicted_probability = np.max(predictions)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_label = labels[predicted_class[0]] if predicted_probability >= 0.5 else "Utility"
    print(f"Predicted Label: {predicted_label}, Probability: {predicted_probability}")

    img = resize_image(img)
    text = perform_ocr(img)

    if predicted_label == "Utility":
        new_label = reclassify_utility(text)
        if new_label != "Utility":
            predicted_label = new_label
            print(f"Reclassified Label: {predicted_label}")

    extract_info(text, predicted_label)

def main(file_path):
    start_time = time.time()

    if file_path.endswith('.pdf'):
        images = process_pdf_and_extract_images(file_path)
    else:
        images = [Image.open(file_path)]

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_single_image, img) for img in images]
        for future in as_completed(futures):
            future.result()  # This will raise any exceptions encountered during processing

    print(f"Time Taken: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    while True:
        file_path = input("Enter the image or PDF file path (or type 'exit' to quit): ")
        if file_path.startswith('"') and file_path.endswith('"'):
            file_path = file_path[1:-1]
        if file_path.lower() == 'exit':
            break
        main(file_path)

 