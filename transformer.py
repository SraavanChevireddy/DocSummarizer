import os
import PyPDF2
from docx import Document
from transformers import pipeline
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=0 if device == "mps" else -1
)

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

def chunk_text(text, max_chunk_size=800):
    sentences = text.split(". ")
    chunks = []
    chunk = ""

    for sentence in sentences:
        if len(chunk) + len(sentence) <= max_chunk_size:
            chunk += sentence + ". "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + ". "

    if chunk:
        chunks.append(chunk.strip())
    return chunks

def summarize_text(text, max_length=200, min_length=100):
    try:
        summary = summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            truncation=True
        )
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return ""

def summarize_document(text):
    chunks = chunk_text(text, max_chunk_size=800)
    combined_summary = ""

    print(f"Text is split into {len(chunks)} chunks for summarization...")
    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i + 1}/{len(chunks)}...")
        combined_summary += summarize_text(chunk) + " "

    print("\nGenerating final summary...")
    final_summary = summarize_text(combined_summary.strip())
    return final_summary

# Main function
def main():
    file_path = input("Enter the path of the medical document (PDF or DOCX): ")

    if not os.path.exists(file_path):
        print("File not found.")
        return

    file_ext = os.path.splitext(file_path)[1].lower()
    text = ""
    if file_ext == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif file_ext == ".docx":
        text = extract_text_from_docx(file_path)
    else:
        print("Unsupported file format.")
        return

    if not text.strip():
        print("No text extracted from the file.")
        return

    print("Summarizing text...")
    summary = summarize_document(text)
    print("\n--- Final Document Summary ---")
    print(summary)

if __name__ == "__main__":
    main()
