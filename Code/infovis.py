import sys
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# Get the student index number.
find_index = sys.argv[1]

# Read the xml file.
tree = ET.parse('info.xml')
root = tree.getroot()

# Filter the student data from the given xml file and store inside a list.
student_data = []

try:
    for attendance_record in root.findall(".//student[index='" + find_index + "']"):
        index = attendance_record.find('index').text
        student_name = attendance_record.find('name').text
        status = attendance_record.find('status').text
        student_data.append({'Index': index, 'Status': status})

    # Calculate the number of total days, present days, and absent days.
    total_days = len(student_data)
    days_present = sum(1 for record in student_data if record['Status'] == 'Present')
    days_absent = sum(1 for record in student_data if record['Status'] == 'Absent')

    # Initializing data for the pie chart.
    labels = ['Present', 'Absent']
    sizes = [days_present, days_absent]
    colors = ['green', 'red']

    # Explode the 'Present' slice for emphasis.
    explode = (0.1, 0)

    # Generate the pie chart.
    plt.figure(figsize=(8, 7))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=False, startangle=120)
    plt.title(f'Attendance Summary for {student_name} - Calculate by Considering {total_days} days.')

    # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.axis('equal')

    # Display the chart.
    plt.show()

except Exception as e:
    print(f"Index number is not available.")
    sys.exit(1)
