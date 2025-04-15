import json
import PyPDF2
import os
import sys

def convert_pdf_to_json(pdf_path, json_path=None):
    """Convert a PDF file to JSON format.
    
    Args:
        pdf_path (str): Path to the PDF file
        json_path (str, optional): Path for the output JSON file. If None, will use the PDF name with .json extension
    
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    try:
        # Validate input file exists
        if not os.path.exists(pdf_path):
            print(f"Error: PDF file not found at {pdf_path}")
            return False
            
        # Determine output path if not provided
        if json_path is None:
            base_name = os.path.splitext(pdf_path)[0]
            json_path = f"{base_name}.json"
        
        print(f"Converting {pdf_path} to JSON...")
        
        # Open the PDF file in binary mode
        with open(pdf_path, 'rb') as file:
            try:
                pdf_reader = PyPDF2.PdfReader(file)
                # Extract text from each page
                text = ""
                total_pages = len(pdf_reader.pages)
                
                print(f"Processing {total_pages} pages...")
                
                for i, page in enumerate(pdf_reader.pages):
                    # Show progress for large documents
                    if i % 10 == 0 and total_pages > 20:
                        print(f"Processing page {i+1}/{total_pages}")
                    
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                    else:
                        print(f"Warning: Could not extract text from page {i+1}")
                
                # Organize the text into a dictionary with metadata
                data = {
                    "title": os.path.basename(pdf_path),
                    "total_pages": total_pages,
                    "content": text
                }
                
                # Convert the dictionary to a JSON string
                json_data = json.dumps(data, indent=4)
                
                # Write the JSON data to a file
                with open(json_path, 'w') as json_file:
                    json_file.write(json_data)
                
                print(f"PDF content has been successfully converted to JSON at {json_path}!")
                return True
                
            except Exception as e:
                print(f"Error reading PDF: {str(e)}")
                return False
    
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False


if __name__ == "__main__":
    # If run as a script, process command line arguments
    if len(sys.argv) < 2:
        print("Usage: python pdf_to_json.py <pdf_path> [json_output_path]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    json_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = convert_pdf_to_json(pdf_path, json_path)
    sys.exit(0 if success else 1)
