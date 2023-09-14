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



