import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# .env file se key load karega
load_dotenv()

print("Testing Google Gemini API...")
print(f"Key loaded: {bool(os.getenv('GOOGLE_API_KEY'))}")

try:
    # Model .env ke GEMINI_MODEL se aata hai; default gemini-3.6-flash.
    # Route through resolve_model_name() so retired names never cause a 404.
    from agent import resolve_model_name
    model_name = resolve_model_name()
    print(f"Model: {model_name}")
    llm = ChatGoogleGenerativeAI(model=model_name) 
    response = llm.invoke("Hello, are you working?")
    print("\n✅ SUCCESS! API is working perfectly.")
    print("AI Response:", response.content)
except Exception as e:
    print("\n❌ FAILED! Asli Error Ye Hai:")
    print(e)