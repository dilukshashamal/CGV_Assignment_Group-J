#sams.py
import cv2
import pytesseract
import xml.etree.ElementTree as ET

# Load the image
image_path = 'img.png'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale Image', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Binarize the image 
binary_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 11)
cv2.imshow('Binarized Image', binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Perform OCR to extract text from the image
extracted_text = pytesseract.image_to_string(binary_image)

# Define the XML content with triple quotes
xml_content = 'info.xml'

# Load attendance data from the XML content
attendance_data = {}
root = ET.parse(xml_content).getroot()


for student in root.findall('.//student'):
    index = student.find('index').text
    name = student.find('name').text
    attendance_data[index] = name

# Process the OCR results to determine attendance
attendance = {}
for line in extracted_text.splitlines():
    for index, name in attendance_data.items():
        if index in line:
            attendance[name] = 'present'
            break
    else:
        attendance[name] = 'absent'

# Display the attendance results with index, name and status
for index, name in attendance_data.items():
    print(index, name, attendance[name])

# Close any open windows
cv2.destroyAllWindows()

#infovis.py ---- not complete ----
import matplotlib.pyplot as plt
import mysql.connector

# Function to fetch student attendance data from the database
def fetch_student_attendance(index_number):
    try:
        # Replace these values with the database connection values for your system
        conn = mysql.connector.connect(host='localhost',
                                       database='student_attendance',
                                       user='root',
                                       password='Fgv#erd2')
        cursor = conn.cursor()

        # Convert index_number to string
        index_number = str(index_number)

        # Fetch student attendance information from the database
        cursor.execute("SELECT student_name, attendance FROM student WHERE index_number = %s", (index_number,))
        record = cursor.fetchone()

        # Close the cursor and the database connection
        cursor.close()
        conn.close()

        return record
    
    except mysql.connector.Error as error:
        print('Error: {}'.format(error))
        return None

# Function to plot the attendance information
def plot_student_attendance(student_name, attendance):
    # Data for the bar graph
    labels = ['Present', 'Absent']
    values = [0, 0] 

    if attendance == 'Present':
        values[0] = 1
    elif attendance == 'Absent':
        values[1] = 1

    # Create the bar graph
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=['green', 'red'])
    plt.xlabel('Attendance Status')
    plt.ylabel('Count')
    plt.title(f'Attendance Summary for {student_name}')
    plt.show()

# Input student index
student_index = input("Enter student index: ")

# Fetch student attendance data from the database
record = fetch_student_attendance(student_index)

if record:
    student_name, attendance = record
    plot_student_attendance(student_name, attendance)
else:
    print(f"Student with index {student_index} not found.")

# create demoinvestigate.py( add machine learning libraries for identify signature
import cv2
import pytesseract
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Function to preprocess the image
def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarize the image
    binary_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 11)

    return binary_image


# Function to extract text using Tesseract OCR
def extract_text(image_path):
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)

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
        if 'Student No' in line:
            student_number = line.split(':')[-1].strip()
            signature_pattern = None
        # Check if the line contains a signature pattern
        elif 'Signature' in line.lower():
            signature_pattern = line.strip()
            student_info[student_number] = signature_pattern
            student_number = None

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
    image_files = ['img.png', 'img1.png', 'img2.png']

    # Process each attendance sheet image
    for image_file in image_files:
        print(f"Processing {image_file}...")
        text = extract_text(image_file)
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



