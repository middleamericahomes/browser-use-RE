import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API credentials
api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_ENDPOINT", "https://api.deepseek.com")

# Display credential info (partially masked for security)
if api_key:
    print(f"API Key: {api_key[:5]}...{api_key[-5:]}")
else:
    print("API Key not found in environment variables")
print(f"Base URL: {base_url}")

# Try both authentication methods
headers_with_bearer = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
headers_direct = {"Authorization": api_key, "Content-Type": "application/json"}

# Simple test payload
payload = {
    "model": "deepseek-chat", 
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."}, 
        {"role": "user", "content": "Say 'DeepSeek API is working!'"}
    ], 
    "stream": False
}

# Test with Bearer token
print("\nTesting with 'Bearer' token...")
try:
    response = requests.post(f"{base_url}/v1/chat/completions", headers=headers_with_bearer, json=payload, timeout=10)
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        print("Success! Response:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Exception: {str(e)}")

# Test with direct API key
print("\nTesting with direct API key...")
try:
    response = requests.post(f"{base_url}/v1/chat/completions", headers=headers_direct, json=payload, timeout=10)
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        print("Success! Response:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Exception: {str(e)}") 