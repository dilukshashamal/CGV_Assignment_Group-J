import cv2
import pytesseract
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Function to preprocess the image
def preprocess_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarize the image
    binary_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 11)

    return binary_image

# Function to extract text using Tesseract OCR
def extract_text(image):
    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Perform OCR to extract text from the image
    extracted_text = pytesseract.image_to_string(preprocessed_image)

    return extracted_text

# Function to extract student numbers and signatures
def extract_student_info(text):
    # Split the text into lines
    lines = text.splitlines()

    # Extract student numbers and signature patterns
    student_info = {}
    student_number = None
    signature_pattern = None

    for line in lines:
        # Check if the line contains a student number
        if 'Student Index' in line:
            student_number = line.split(':')[-1].strip()
            signature_pattern = None
        # Check if the line contains a signature pattern
        elif 'Signature' in line:
            signature_pattern = line.strip()
            student_info[student_number] = signature_pattern

    return student_info

# Function to compare signatures using cosine similarity
def compare_signatures(signature1, signature2):
    features1 = np.array(signature1).reshape(1, -1)
    features2 = np.array(signature2).reshape(1, -1)
    similarity_score = cosine_similarity(features1, features2)
    return similarity_score[0][0]

# Main script
if __name__ == "__main__":
    # List of attendance sheet image file names
    image_files = ['1.JPEG', '2.JPEG', '3.JPEG']

    # Process each attendance sheet image
    for image_file in image_files:
        print(f"Processing {image_file}...")
        
        # Read the image
        image = cv2.imread(image_file)
        
        # Show the original image
        cv2.imshow('Original Image', image)
        cv2.waitKey(0)
        
        text = extract_text(image)
        student_info = extract_student_info(text)

        # Extract reference signature pattern
        reference_student_number = '001'  # Replace with the actual student number
        reference_signature_pattern = student_info.get(reference_student_number)

        # Print extracted student numbers and signature patterns
        for student_number, signature_pattern in student_info.items():
            print(f"Student Number: {student_number}")
            print(f"Signature Pattern: {signature_pattern}")

            # Compare signatures if a reference signature pattern is available
            if reference_signature_pattern:
                similarity_score = compare_signatures(reference_signature_pattern, signature_pattern)
                similarity_threshold = 0.7  # Adjust the threshold as needed
                if similarity_score >= similarity_threshold:
                    print(f"Signatures match with a similarity score of {similarity_score:.2f}")
                else:
                    print(f"Signatures do not match with a similarity score of {similarity_score:.2f}")

            print("\n")
            
        # Close the image window
        cv2.destroyAllWindows()



