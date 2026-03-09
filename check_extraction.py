import os
import time
import requests
import html
import re
from dotenv import load_dotenv

def convert_markdown_tables_to_text(markdown_text: str) -> str:
    """
    Convert markdown tables to natural language sentences cleanly.
    Handles both standard tables and 2-column Key-Value tables.
    """
    # FIXED REGEX: Matches any block of consecutive lines that start and end with a pipe |
    table_pattern = r'(?m)(^[ \t]*\|[^\n]*\|[ \t]*\n(?:^[ \t]*\|[^\n]*\|[ \t]*\n?)*)'
    
    def convert_single_table(table_match):
        table_text = table_match.group(0)
        lines = [line.strip() for line in table_text.strip().split('\n')]
        
        sentences = []
        for line in lines:
            # Skip empty lines or Markdown separator lines (e.g., |---|---| )
            if not line or re.match(r'^\|[\s\-\|:]+\|$', line):
                continue
                
            # Extract the actual text cells from the row
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            cells = [c for c in cells if c] # Remove empty cells
            
            if cells:
                # If it's a 2-column table, format as "Key: Value."
                if len(cells) == 2:
                    sentences.append(f"{cells[0]}: {cells[1]}.")
                # If it's a multi-column table, format as comma-separated values
                else:
                    sentences.append(", ".join(cells) + ".")
                    
        if sentences:
            return "\n\n[CONVERTED TABLE DATA]:\n" + " ".join(sentences) + "\n\n"
        return table_text

    # Apply the conversion to the entire document
    converted = re.sub(table_pattern, convert_single_table, markdown_text)
    return converted

# =========================================================================
# MAIN SCRIPT
# =========================================================================

load_dotenv()
api_key = os.getenv("LLAMA_CLOUD_API_KEY")
base_url = "https://api.cloud.llamaindex.ai/api/parsing"

headers = {
    "Authorization": f"Bearer {api_key}"
}

file_path = "222-rajpur-brochure.pdf"

with open(file_path, "rb") as f:
    files = {"file": (file_path, f, "application/pdf")}
    data = {"language": "en"}
    
    print("Uploading to LlamaParse REST API...")
    upload_response = requests.post(f"{base_url}/upload", headers=headers, files=files, data=data, verify=False)

if upload_response.status_code != 200:
    print(f"Upload failed! HTTP {upload_response.status_code}")
    print(upload_response.text)
    exit()

job_id = upload_response.json()["id"]
print(f"Upload successful. Job ID: {job_id}")

while True:
    status_response = requests.get(f"{base_url}/job/{job_id}", headers=headers, verify=False)
    status_data = status_response.json()
    
    if status_data["status"] == "SUCCESS":
        print("Parsing complete!")
        break
    elif status_data["status"] == "ERROR":
        print("LlamaParse server failed to process the document.")
        print(status_data)
        exit()
        
    print("Still processing... waiting 5 seconds.")
    time.sleep(5)

result_response = requests.get(f"{base_url}/job/{job_id}/result/markdown", headers=headers, verify=False)

# 1. Parse the JSON response
response_data = result_response.json()
raw_markdown = response_data.get("markdown", "")

# 2. Decode HTML symbols
clean_markdown = html.unescape(raw_markdown)

# 3. Write ORIGINAL markdown to file
with open("extracted_output.txt", "w", encoding="utf-8") as f:
    f.write(clean_markdown)
print("1️⃣ Original markdown saved to: extracted_output.txt")

# 4. Write CONVERTED markdown to file
converted_markdown = convert_markdown_tables_to_text(clean_markdown)
with open("extracted_output_1.txt", "w", encoding="utf-8") as f:
    f.write(converted_markdown)
print("2️⃣ Table-converted markdown saved to: extracted_output_1.txt")


# =========================================================================
# NEW: CHUNKING PREVIEW (See the Markdown Strategy in action!)
# =========================================================================
print("\n" + "="*50)
print("🔬 CHUNKING PREVIEW (First 3 Chunks)")
print("="*50)

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# Initialize splitters
markdown_headers = [
    ("#", "Header1"),
    ("##", "Header2"),
    ("###", "Header3"),
]
md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=markdown_headers, strip_headers=False)
char_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)

# Process the converted text
md_docs = md_splitter.split_text(converted_markdown)
final_docs = []

for doc in md_docs:
    if len(doc.page_content) > 1200:
        sub_texts = char_splitter.split_text(doc.page_content)
        for text in sub_texts:
            final_docs.append({"content": text, "metadata": doc.metadata})
    else:
        final_docs.append({"content": doc.page_content, "metadata": doc.metadata})

# Print the first 3 chunks to verify headers are attached
for i, doc in enumerate(final_docs[:3]):
    print(f"\n--- CHUNK {i+1} ---")
    print(f"🏷️  METADATA: {doc['metadata']}")
    print(f"📝 CONTENT:\n{doc['content'][:200]}...") # Print first 200 chars