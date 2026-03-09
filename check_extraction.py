import os
import time
import requests
import html
from dotenv import load_dotenv

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
    upload_response = requests.post(f"{base_url}/upload", headers=headers, files=files, data=data,verify=False)

if upload_response.status_code != 200:
    print(f"Upload failed! HTTP {upload_response.status_code}")
    print(upload_response.text)
    exit()

job_id = upload_response.json()["id"]
print(f"Upload successful. Job ID: {job_id}")

while True:
    status_response = requests.get(f"{base_url}/job/{job_id}", headers=headers,verify=False)
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

'''result_response = requests.get(f"{base_url}/job/{job_id}/result/markdown", headers=headers,verify=False)

with open("extracted_output.txt", "w", encoding="utf-8") as f:
    f.write(result_response.text)
    
print("Raw markdown saved to extracted_output.txt")'''

result_response = requests.get(f"{base_url}/job/{job_id}/result/markdown", headers=headers, verify=False)

# 1. Parse the JSON response to extract just the markdown string
response_data = result_response.json()
raw_markdown = response_data.get("markdown", "")

# 2. Decode nasty HTML symbols (like &#x26; into &)
clean_markdown = html.unescape(raw_markdown)

# 3. Write it to the file so the \n become actual new lines
with open("extracted_output.txt", "w", encoding="utf-8") as f:
    f.write(clean_markdown)
    
print("Clean, formatted markdown saved to extracted_output.txt")