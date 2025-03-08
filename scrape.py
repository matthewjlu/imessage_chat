import os
import shutil
import subprocess
from bs4 import BeautifulSoup

folder = '/Users/mattlu/imessage_export'

# Iterate over all the items in the folder and remove them
for entry in os.listdir(folder):
    entry_path = os.path.join(folder, entry)
    try:
        if os.path.isfile(entry_path) or os.path.islink(entry_path):
            os.unlink(entry_path)  # remove file or link
        elif os.path.isdir(entry_path):
            shutil.rmtree(entry_path)  # remove directory and its contents
    except Exception as e:
        print(f'Failed to delete {entry_path}. Reason: {e}')

print("Folder cleared.")

# Run the imessage-exporter command
contact = input("Enter the contact's phone number: ")
command = 'imessage-exporter -f html -t "' + contact + '" '
result = subprocess.run(command, shell=True, capture_output=True, text=True)

# Optionally print the command's output
print(result.stdout)
if result.stderr:
    print("Error:", result.stderr)

# Grab a certain file from the folder and add it to your directory, then rename it
target_file = "+1" + contact + ".html"  # Source filename
source_path = os.path.join(folder, target_file)
destination_path = os.path.join(os.getcwd(), target_file)
name = target_file

if os.path.exists(source_path) and os.path.isfile(source_path):
    shutil.copy2(source_path, destination_path)
    print(f"{target_file} has been added to your directory.")
    
    new_name = input("Enter a new name for the file (including .html) or press Enter to keep the same: ")

    if new_name:
        name = new_name
        new_destination_path = os.path.join(os.getcwd(), new_name)
        os.rename(destination_path, new_destination_path)
        print(f"File has been renamed to {new_name}.")
else:
    print(f"File not found: {source_path}")

# Open and parse the HTML file
with open(name + '.html', 'r', encoding='utf-8') as file:
    soup = BeautifulSoup(file, 'html.parser')

# Find all message blocks with the class "received" (which are displayed in gray)
message_type = input("Enter the message type (received/sent iMessage): ")

received_messages = soup.find_all('div', class_=message_type)

# Extract text from the spans with class "bubble" (where the actual message text is)
texts = []
for msg in received_messages:
    bubbles = msg.find_all('span', class_='bubble')
    for bubble in bubbles:
        text = bubble.get_text(strip=True)
        if text:  # Only add non-empty texts
            texts.append(text)

# Save the extracted texts to a txt file
with open('received_texts.csv', 'w', encoding='utf-8') as out_file:
    for text in texts:
        out_file.write(text + "\n")

print("Received messages saved to received_texts.csv")
