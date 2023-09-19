import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pytesseract

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 11)

    return binary_image

# Function to extract signatures and student index from an image
def extract_signatures_and_index(image_path):
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)

    extracted_text = pytesseract.image_to_string(preprocessed_image)
    contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    signatures = []

    # Iterate through the detected contours
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            signature = preprocessed_image[y:y+h, x:x+w]
            signature = cv2.resize(signature, (200, 200))
            signatures.append((signature, extracted_text))

    return signatures

# Function to extract signatures from an image
def extract_signatures(image_path):
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)

    # Find contours in the binary image
    contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    signatures = []

    # Iterate through the detected contours
    for contour in contours:
        # Filter out small contours 
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            # Crop the signature region
            signature = preprocessed_image[y:y+h, x:x+w]
            # Resize all signatures to a common size 
            signature = cv2.resize(signature, (200, 200))
            signatures.append(signature)

    return signatures

# Function to compare signatures using cosine similarity
def compare_signatures(signature1, signature2):
    features1 = np.array(signature1).reshape(1, -1)
    features2 = np.array(signature2).reshape(1, -1)
    similarity_score = cosine_similarity(features1, features2)
    return similarity_score[0][0]

if __name__ == "__main__":
    image_files = ['1.jpeg', '2.jpeg', '3.jpeg']
    signatures_by_student_index = {}

    for image_file in image_files:
        print(f"Processing {image_file}...")  
        # Extract signatures and associated student index from the image
        signatures_with_index = extract_signatures_and_index(image_file)

        # Group signatures by student index
        for signature, student_index in signatures_with_index:
            if student_index not in signatures_by_student_index:
                signatures_by_student_index[student_index] = []
            signatures_by_student_index[student_index].append(signature)

    # Compare signatures within each student index group
    for student_index, student_signatures in signatures_by_student_index.items():
        print(f"Student Index: {student_index}")
        for i in range(len(student_signatures)):
            for j in range(i+1, len(student_signatures)):
                similarity_score = compare_signatures(student_signatures[i], student_signatures[j])
                print(f"Similarity between Signature {i+1} and Signature {j+1}: {similarity_score:.2f}")

                # Display the cropped signatures
                cv2.imshow(f"Signature {i+1}", student_signatures[i])
                cv2.imshow(f"Signature {j+1}", student_signatures[j])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                print()
        print()

    # Compare signatures across all student index groups
    student_indices = list(signatures_by_student_index.keys())
    
    for i in range(len(student_indices)):

        for j in range(i+1, len(student_indices)):
            student_index1 = student_indices[i]
            student_index2 = student_indices[j]

            for signature1 in signatures_by_student_index[student_index1]:
                for signature2 in signatures_by_student_index[student_index2]:
                    similarity_score = compare_signatures(signature1, signature2)
                    print(f"Similarity between Signature of Student {student_index1} and Signature of Student {student_index2}: {similarity_score:.2f}")

                    # Display the cropped signatures
                    cv2.imshow(f"Signature of Student {student_index1}", signature1)
                    cv2.imshow(f"Signature of Student {student_index2}", signature2)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    print()
        print()


