'''Check that the API keys in .env work: sends one short test message to each configured LLM.

Run from the project folder:   python tests/check_llm_apis.py
'''
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

if hasattr(sys.stdout, 'reconfigure'):  # never crash on old Windows console encodings
    sys.stdout.reconfigure(errors='replace')

load_dotenv(Path(__file__).resolve().parents[1] / '.env')

from llm import build_chat_model, configured_providers, describe_error


def main() -> int:
    providers = configured_providers()
    if not providers:
        print('No API key found. Put GOOGLE_API_KEY and/or GROQ_API_KEY into the .env file.')
        return 1
    failures = 0
    for provider in providers:
        print(f'Testing {provider.label} (model: {provider.model}) ...')
        try:
            reply = build_chat_model(provider, timeout=30).invoke('Say hello in five words.')
            print(f'  [OK] WORKS - reply: {str(reply.text).strip()}')  # .text is a property (not a method call)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f'  [FAILED] {describe_error(exc)}')
    return 1 if failures == len(providers) else 0


if __name__ == '__main__':
    sys.exit(main())
