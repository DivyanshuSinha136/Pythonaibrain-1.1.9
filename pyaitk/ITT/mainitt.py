import easyocr

# Create the reader once at the top level
reader = easyocr.Reader(['en'])  # Download model files once

def ITT(image_path, languages=['en']):
    result = reader.readtext(image_path)
    extracted_text = " ".join([text for _, text, _ in result])
    return extracted_text
