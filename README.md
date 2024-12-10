**Currently, I am only running this project on the command prompt. You can prepare a good UI. For the completely trained model supporting all 16 entities, contact me via [LinkedIn](https://www.linkedin.com/in/dhruv-juneja-b6599a1a9).**  

**Note**:  
For certain countries like South Africa, Spain, and others, sample identification cards were created using templates, as direct access to real-world samples was not available. These templates were used solely for training purposes.  

# Intelligent Document Classification and PII Extraction (OCR & NER)  

This repository provides a script for document classification and Optical Character Recognition (OCR) for various types of identification documents. It supports classification using a Convolutional Neural Network (CNN) model and extracts key information such as Aadhaar numbers, PAN numbers, Passport numbers, and more using OCR and regex patterns.  

## Project Overview  

This is a project for a leading software company that aims to solve the problem of securely managing Personally Identifiable Information (PII). When an image is uploaded, the system identifies whether it contains critical identity information (e.g., ID cards, passports) and blocks its sharing, as storing such numbers is not permitted by the government.  

To train the model, datasets were sourced from platforms like Kaggle. I am not responsible for any information regarding the personal data on these cards.  

## How It Works  

### Key Features  
1. **Document Classification**  
   The script uses a pre-trained CNN model to classify uploaded images into categories like Aadhaar, Passport, Driving License, and more.  

2. **OCR for Text Extraction**  
   Text is extracted from images, PDFs, DOCX, or TXT files using Tesseract OCR. The OCR engine is configured to support multiple languages (English, Hindi, Marathi).  

3. **Information Extraction**  
   Key details like Aadhaar numbers, PAN numbers, and Passport numbers are identified using regular expressions tailored for each document type.  

4. **Utility Reclassification**  
   If the classification confidence is low, the script reclassifies the document type based on keywords in the extracted text.  

5. **PDF, DOCX, TXT, and Image Support**  
   The script can process single images, extract multiple images from PDFs, and work with text-based files (DOCX, TXT) for identifying PII.  

6. **Multi-threaded Processing**  
   The script uses multi-threading to speed up the processing of multiple pages or images.  

### Supported Document Types  
- **Aadhaar (India)**  
  Extracts Aadhaar numbers in various formats (English, Hindi, Tamil, etc.).  

- **PAN (India)**  
  Recognizes Indian PAN card numbers.  

- **Passports**  
  Supports multiple passport formats, including India and the US.  

- **Driving Licenses**  
  Recognizes driving license numbers from the UK and India.  

- **Voter ID (India)**  
  Extracts Indian voter identification numbers.  

- **Canada ID**  
  Identifies Canadian ID numbers.  

- **Czech Republic ID**  
  Recognizes IDs from the Czech Republic.  

- **Denmark Personal Identification Number (PID)**  
  Extracts 10-digit Danish personal identification numbers.  

- **Finland ID**  
  Recognizes Finnish identification numbers.  

- **Israel National ID (NID)**  
  Extracts 9-digit Israeli national identification numbers.  

- **Poland ID**  
  Recognizes Polish ID numbers.  

- **Singapore ID**  
  Extracts Singaporean identification numbers.  

- **South Africa ID**  
  Identifies 13-digit South African ID numbers.  

- **Spain Social Security Numbers (SSN)**  
  Recognizes Spanish SSNs in the format "12345678-A."  

- **Utility Documents**  
  Reclassification of documents if they do not match predefined categories.  
