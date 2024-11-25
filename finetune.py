import os
import json
from PyPDF2 import PdfReader

# Function to extract text from PDFs and format it into a dataset for fine-tuning
def extract_pdf_text_for_finetuning(pdf_files, output_file="fine_tuning_dataset.jsonl"):
    fine_tuning_data = []

    # Iterate through each PDF and extract text
    for pdf in pdf_files:
        try:
            pdf_reader = PdfReader(pdf)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    # Create a prompt-completion pair (for simplicity, page text is both the prompt and completion)
                    prompt = f"Page {page_num + 1} content of {os.path.basename(pdf)}"
                    completion = page_text.strip().replace("\n", " ")
                    fine_tuning_data.append({"prompt": prompt, "completion": completion})
            print(f"Successfully processed: {pdf}")
        except Exception as e:
            print(f"Error reading {pdf}: {e}")
            continue

    # Save the extracted content to a JSONL file for fine-tuning
    with open(output_file, "w") as outfile:
        for entry in fine_tuning_data:
            json.dump(entry, outfile)
            outfile.write("\n")  # JSONL format requires new lines between records

    print(f"Fine-tuning dataset saved to {output_file}")

if __name__ == "__main__":
    # Define folder where PDFs are stored
    folder_path = "data"  # The 'data' folder should contain your PDFs

    # Get all PDF files in the folder
    pdf_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".pdf")]

    if not pdf_files:
        print("No PDF files found in the folder.")
    else:
        # Extract text and format it for fine-tuning
        extract_pdf_text_for_finetuning(pdf_files, output_file="fine_tuning_dataset.jsonl")
