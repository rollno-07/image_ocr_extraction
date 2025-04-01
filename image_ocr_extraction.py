import os
import re
import cv2
import json
import pytesseract
from dotenv import load_dotenv
import google.generativeai as genai
import autogen
from pdf2image import convert_from_path

load_dotenv()

# Read API key from environment variables
api_key = os.getenv("GEMINI_API_KEY")
# 🔹 Configure Google Gemini API (Replace with your key)
genai.configure(api_key=api_key)

# 🔹 Function to Extract Text from Images
def extract_text_from_image(image_path):
    """Extract text from an image using Tesseract OCR."""
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Image file not found or cannot be read."

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    text = pytesseract.image_to_string(gray, lang="eng")  # Perform OCR
    return text.strip()

# 🔹 Function to Extract Text from PDFs
def extract_text_from_pdf(pdf_path):
    """Convert PDF pages to images and extract text using OCR."""
    try:
        images = convert_from_path(pdf_path)
        text = ""
        for image in images:
            text += pytesseract.image_to_string(image, lang="eng") + "\n"
        return text.strip()
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

# 🔹 Function to Extract Structured Data with Gemini LLM
def extract_structured_data(raw_text):
    prompt = f"""
    Extract the following details from the given text and return in JSON format:
    - Name
    - PAN Number
    - Date of Birth (DOB)
    
    
    Raw Text:
    {raw_text}

    Format the response strictly as JSON without any extra text.
    Example:
    {{
        "Name": "John Doe",
        "PAN": "ABCDE1234F",
        "DOB": "01/01/1990",
    }}
    """

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)

        if not response or not response.text.strip():
            return {"error": "Gemini API returned an empty response"}

        # Extract JSON using regex (in case Gemini returns extra text)
        json_match = re.search(r"\{.*\}", response.text, re.DOTALL)

        if not json_match:
            return {"error": "Gemini API returned invalid JSON format"}

        extracted_json = json_match.group(0)  # Get matched JSON string

        try:
            extracted_data = json.loads(extracted_json)  # Parse JSON output
        except json.JSONDecodeError:
            return {"error": "Gemini API returned malformed JSON"}

        return extracted_data

    except Exception as e:
        return {"error": f"Gemini API error: {str(e)}"}
# 🔹 Function to Validate Extracted Data with Autogen


def validate_extracted_data(data):
    print(data, "Debug: Initial extracted data")  # Debugging output

    """Autogen agent validates extracted JSON and ensures correctness."""
    
    assistant = autogen.AssistantAgent(
        name="validator",
        system_message=(
            "You are a JSON validation assistant. "
            "Your ONLY task is to check the JSON and return a valid JSON object. "
            "You MUST respond ONLY with JSON and nothing else. "
            "DO NOT add explanations, instructions, or extra text. "
            "If the input JSON is correct, return it exactly as it is. "
            "If incorrect, return a corrected version."
        )
    )

    user_proxy = autogen.UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        is_termination_msg=lambda x: x.get("content") in ["TERMINATE", "DONE"],
        code_execution_config={"use_docker": False}
    )

    validation_prompt = f"""{json.dumps(data, indent=4)}"""

    try:
        response = user_proxy.initiate_chat(assistant, message=validation_prompt)

        # Debugging: Print the entire response structure
        print("Raw Response:", response)

        # Extract assistant's response
        extracted_content = None
        if hasattr(response, "chat_history") and response.chat_history:
            for message in response.chat_history:
                if message["role"] == "assistant":
                    extracted_content = message.get("content", "").strip()
                    if extracted_content:
                        break  # Stop at first valid assistant response

        if not extracted_content:
            print("❌ No valid content returned from assistant")
            return {"error": "Assistant did not return a valid JSON response."}

        print("Extracted Content:", extracted_content)  # Debugging step
        
        # Try parsing JSON, handle cases where response is incorrect
        try:
            return json.loads(extracted_content)
        except json.JSONDecodeError as e:
            print("❌ JSON Decoding Failed:", e)
            return {"error": f"Invalid JSON response: {str(e)}", "raw_response": extracted_content}

    except Exception as e:
        return {"error": f"Validation failed: {str(e)}"}

# 🔹 Main Function
def main(file_path):
    """Main function to process the file and extract structured JSON data."""
    if not os.path.exists(file_path):
        print(json.dumps({"error": "File not found!"}, indent=4))
        return

    # Detect file type and extract raw text
    if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
        raw_text = extract_text_from_image(file_path)
    elif file_path.lower().endswith(".pdf"):
        raw_text = extract_text_from_pdf(file_path)
    else:
        print(json.dumps({"error": "Unsupported file format!"}, indent=4))
        return

    print("\n🔹 Extracted Raw Text:\n", raw_text)  # Debugging output

    # Step 2: Extract structured data using LLM
    structured_data = extract_structured_data(raw_text)

    # Step 3: Validate the extracted data using Autogen
    validated_data = validate_extracted_data(structured_data)

    # Print the final validated JSON output
    print("\n✅ Final Extracted Data:\n", json.dumps(validated_data, indent=4))

# 🔹 Run the script
if __name__ == "__main__":
    main("Your doc file path")  # Replace with actual file path


