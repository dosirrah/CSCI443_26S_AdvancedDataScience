import json
import base64
import sys

# Load your notebook
with open(sys.argv[1], "r", encoding="utf-8") as f:
    nb = json.load(f)

# Extract and save image attachments
image_count = 0
for cell in nb['cells']:
    if 'attachments' in cell:
        for name, attachment in cell['attachments'].items():
            for mime_type, base64_data in attachment.items():
                # Determine file extension
                ext = mime_type.split("/")[-1]
                filename = f"extracted_image_{image_count}.{ext}"
                with open(filename, "wb") as img_file:
                    img_file.write(base64.b64decode(base64_data))
                print(f"Saved {filename}")
                image_count += 1
