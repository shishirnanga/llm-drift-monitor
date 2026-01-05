"""Quick test to verify setup is working"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check API keys exist
openai_key = os.getenv("OPENAI_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")

print("=" * 50)
print("SETUP VERIFICATION")
print("=" * 50)

# Check OpenAI
if openai_key and openai_key != "sk-your-openai-key-here":
    print("✅ OpenAI API key found")
    
    # Try to import and create client
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        print("✅ OpenAI client created successfully")
    except Exception as e:
        print(f"❌ OpenAI error: {e}")
else:
    print("❌ OpenAI API key missing or not set")

# Check Anthropic
if anthropic_key and anthropic_key != "sk-ant-your-anthropic-key-here":
    print("✅ Anthropic API key found")
    
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=anthropic_key)
        print("✅ Anthropic client created successfully")
    except Exception as e:
        print(f"❌ Anthropic error: {e}")
else:
    print("❌ Anthropic API key missing or not set")

# Check data directories
for dir_path in ["data/raw", "data/processed"]:
    if os.path.exists(dir_path):
        print(f"✅ Directory exists: {dir_path}")
    else:
        print(f"❌ Missing directory: {dir_path}")

print("=" * 50)
print("Setup verification complete!")