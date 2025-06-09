import os
import PyPDF2
from docx import Document
from openai import OpenAI

client = OpenAI(api_key="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")

# **** Replace this placeholder with your actual OpenAI API Key ****

# Function to extract text from PDF files
def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
    except Exception as e:
        print(f"Error reading PDF file: {e}")
    return text

# Function to extract text from DOCX files
def extract_text_from_docx(file_path):
    text = ""
    try:
        doc = Document(file_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        print(f"Error reading DOCX file: {e}")
    return text

# Function to summarize text using OpenAI API
def summarize_text_with_openai(text, max_length=150):
    try:
        response = client.chat.completions.create(model="gpt-4o-mini",  # You can use 'gpt-3.5-turbo' for a cheaper alternative
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text."},
            {"role": "user", "content": f"Please summarize the following text:\n\n{text}"}
        ],
        temperature=0.7,
        max_tokens=max_length)
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error using OpenAI API for summarization: {e}")
        return ""

# Main function to handle input and processing
def main():
    file_path = input("Enter the path of the file you want to summarize (PDF, DOC, or DOCX): ")

    if not os.path.exists(file_path):
        print("Invalid file path.")
        return

    file_ext = os.path.splitext(file_path)[1].lower()
    text = ""
    if file_ext == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif file_ext in [".doc", ".docx"]:
        text = extract_text_from_docx(file_path)
    else:
        print("Unsupported file type. Please provide a PDF, DOC, or DOCX file.")
        return

    if not text.strip():
        print("No text found in the file.")
        return

    print("\n--- Summary ---")
    # Using OpenAI API for summarization
    summary = summarize_text_with_openai(text, max_length=200)  # Adjust the token limit as needed
    print(summary)

if __name__ == "__main__":
    main()

# /Users/sraavanchevireddy/Downloads/My_Resume.pdf
# /Users/sraavanchevireddy/Desktop/Sample-filled-in-MR.pdf