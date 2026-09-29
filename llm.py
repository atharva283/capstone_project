'''LLM providers: Gemini and Groq (each with a backup model), plus friendly error messages.'''
from __future__ import annotations
import logging
import os
import re
from dataclasses import dataclass, field
from langchain_core.language_models import BaseChatModel

log = logging.getLogger('docassistant')
# The Google SDK prints a harmless "automatic function calling" notice on every request.
for _name in ('google_genai', 'google_genai.models', 'google.genai'):
    logging.getLogger(_name).setLevel(logging.ERROR)

# Single source of truth for model names. Providers retire models regularly and a retired name
# answers "model not found", so a stale .env value is silently replaced by the current default.
DEFAULT_GEMINI_MODEL = 'gemini-3.6-flash'
DEFAULT_GEMINI_BACKUP_MODEL = 'gemini-3.5-flash-lite'   # same key, separate quota
DEFAULT_GROQ_MODEL = 'qwen/qwen3.8-27b'
DEFAULT_GROQ_BACKUP_MODEL = 'openai/gpt-oss-20b'        # same key, separate rate limits
DEPRECATED_GEMINI_MODELS = {
    'gemini-2.5-flash', 'gemini-2.5-pro',
    'gemini-1.5-flash', 'gemini-1.5-flash-001', 'gemini-1.5-pro', 'gemini-pro',
}
DEPRECATED_GROQ_MODELS = {   # all of these now answer "model not found" on Groq
    'llama-3.3-70b-versatile', 'llama-3.1-70b-versatile', 'llama-3.1-8b-instant',
    'llama3-70b-8192', 'llama3-8b-8192', 'gemma2-9b-it', 'gemma-7b-it',
    'mixtral-8x7b-32768', 'deepseek-r1-distill-llama-70b',
    # Exists, but failed 9 of 13 test questions: it emits a "commentary" tool call that Groq rejects.
    'openai/gpt-oss-120b',
}
_PLACEHOLDER_HINTS = ('your_', 'paste_', 'placeholder', 'changeme', 'xxxx')


def is_real_key(key: str | None) -> bool:
    '''True when a value looks like a real API key rather than empty text or a template.'''
    value = (key or '').strip().lower()
    return bool(value) and not any(hint in value for hint in _PLACEHOLDER_HINTS)


def _resolve(candidate: str | None, env_name: str, default: str, retired: set[str]) -> str:
    value = (candidate or os.getenv(env_name) or '').strip()
    return default if not value or value in retired else value


def resolve_gemini_model(model_name: str | None = None) -> str:
    '''Explicit argument > GEMINI_MODEL env > default. Blank or retired names use the default.'''
    return _resolve(model_name, 'GEMINI_MODEL', DEFAULT_GEMINI_MODEL, DEPRECATED_GEMINI_MODELS)


def resolve_gemini_backup_model() -> str:
    '''GEMINI_BACKUP_MODEL env > default.'''
    return _resolve(None, 'GEMINI_BACKUP_MODEL', DEFAULT_GEMINI_BACKUP_MODEL, DEPRECATED_GEMINI_MODELS)


def resolve_groq_model(model_name: str | None = None) -> str:
    '''Explicit argument > GROQ_MODEL env > default. Blank or retired names use the default.'''
    return _resolve(model_name, 'GROQ_MODEL', DEFAULT_GROQ_MODEL, DEPRECATED_GROQ_MODELS)


def resolve_groq_backup_model() -> str:
    '''GROQ_BACKUP_MODEL env > default. Blank or retired names use the default.'''
    return _resolve(None, 'GROQ_BACKUP_MODEL', DEFAULT_GROQ_BACKUP_MODEL, DEPRECATED_GROQ_MODELS)


@dataclass(frozen=True)
class Provider:
    '''One configured LLM (a service plus a model). The key is never shown in repr() or logs.'''
    name: str     # 'gemini', 'groq', 'groq-backup' or 'gemini-backup'
    label: str    # human-friendly name for the UI
    model: str
    api_key: str = field(repr=False)


def configured_providers(gemini_key: str | None = None, groq_key: str | None = None) -> list[Provider]:
    '''Every LLM that has a usable key, in the order they are tried:

        Gemini -> Groq -> Groq backup model -> Gemini backup model

    The two main models come first (a different company is the best spare), then each company's
    backup model (same key, separate quota). PRIMARY_LLM=groq in .env swaps the first two.
    Keys passed in (for example from the sidebar) win over the environment / .env file.
    '''
    gemini_key = gemini_key if is_real_key(gemini_key) else os.getenv('GOOGLE_API_KEY')
    groq_key = groq_key if is_real_key(groq_key) else os.getenv('GROQ_API_KEY')
    gemini = groq = gemini_backup = groq_backup = None
    if is_real_key(gemini_key):
        gemini = Provider('gemini', 'Gemini', resolve_gemini_model(), gemini_key.strip())
        if resolve_gemini_backup_model() != gemini.model:
            gemini_backup = Provider('gemini-backup', 'Gemini backup', resolve_gemini_backup_model(), gemini.api_key)
    if is_real_key(groq_key):
        groq = Provider('groq', 'Groq', resolve_groq_model(), groq_key.strip())
        if resolve_groq_backup_model() != groq.model:
            groq_backup = Provider('groq-backup', 'Groq backup', resolve_groq_backup_model(), groq.api_key)
    mains = [groq, gemini] if (os.getenv('PRIMARY_LLM') or '').strip().lower() == 'groq' else [gemini, groq]
    return [provider for provider in (*mains, groq_backup, gemini_backup) if provider]


def build_chat_model(provider: Provider, timeout: int = 45, max_retries: int = 1) -> BaseChatModel:
    '''Create the LangChain chat model for a provider (imports are lazy so unused SDKs never load).'''
    if provider.name.startswith('gemini'):
        from langchain_google_genai import ChatGoogleGenerativeAI
        # No temperature: Gemini 3.x uses fixed sampling defaults and warns if one is passed.
        return ChatGoogleGenerativeAI(
            model=provider.model, google_api_key=provider.api_key,
            max_output_tokens=4096, timeout=timeout, max_retries=max_retries,
        )
    if provider.name.startswith('groq'):
        from langchain_groq import ChatGroq
        # These models "think" before answering. Thinking tokens count against Groq's small
        # per-minute token limit and slow every answer down, so it is switched off / kept low.
        extra = {}
        if 'gpt-oss' in provider.model:
            extra = {'reasoning_effort': 'low'}
        elif 'qwen' in provider.model:
            extra = {'reasoning_effort': 'none'}
        return ChatGroq(
            model=provider.model, api_key=provider.api_key, temperature=0.1,
            max_tokens=4096, timeout=timeout, max_retries=max_retries, **extra,
        )
    raise ValueError(f'Unknown LLM provider: {provider.name}')


def redact(text: str, secrets: list[str]) -> str:
    '''Remove any API key that an upstream error message might echo back.'''
    for secret in secrets:
        if secret and len(secret) > 6:
            text = text.replace(secret, '***')
    return text


def _text(exc: BaseException) -> str:
    return f'{type(exc).__name__} {exc}'.lower()


def _has_code(text: str, *codes: int) -> bool:
    '''True if the text contains an HTTP status code as a whole number (not inside "4013" or "0.401").'''
    return any(re.search(rf'(?<![\d.]){code}(?![\d.])', text) for code in codes)


def is_quota_error(exc: BaseException) -> bool:
    '''True for "quota / rate limit" failures.'''
    text = _text(exc)
    return _has_code(text, 429) or any(word in text for word in (
        'resource_exhausted', 'quota', 'rate limit', 'rate_limit', 'too many requests'))


def is_malformed_tool_call(exc: BaseException) -> bool:
    '''True when the model produced an unusable tool call (a known glitch of some open models).'''
    text = _text(exc)
    return any(word in text for word in ('tool_use_failed', 'tool call validation failed',
                                         'failed to call a function', 'failed_generation'))


def is_daily_limit(exc: BaseException) -> bool:
    '''True when the limit that was hit resets per day (waiting a minute will not help).'''
    text = _text(exc)
    return any(word in text for word in ('perday', 'per day', 'requests per day', 'tokens per day',
                                         '(rpd)', '(tpd)', 'daily'))


_RETRY_HINT = re.compile(r"(?:try again in|retry in|retry after|retrydelay['\": ]+)\s*([0-9hms.]+)", re.I)
_DURATION_PART = re.compile(r'(\d+(?:\.\d+)?)(ms|h|m|s)')


def retry_delay_seconds(exc: BaseException) -> float | None:
    '''Seconds the provider asked us to wait ("try again in 1m3.5s", "retry in 51s"), if it said.'''
    match = _RETRY_HINT.search(str(exc))
    if not match:
        return None
    factor = {'ms': 0.001, 's': 1, 'm': 60, 'h': 3600}
    parts = _DURATION_PART.findall(match.group(1).lower())
    return sum(float(number) * factor[unit] for number, unit in parts) if parts else None


def describe_error(exc: BaseException) -> str:
    '''Turn an upstream exception into a short, safe explanation (never the raw message).'''
    text = _text(exc)
    if _has_code(text, 401, 403) or any(word in text for word in (
            'api key not valid', 'api_key_invalid', 'invalid api key', 'invalid_api_key',
            'unauthorized', 'permission_denied', 'authentication')):
        return 'the API key was rejected (invalid, expired or not allowed)'
    if is_quota_error(exc):
        return 'the free quota or rate limit has been reached'
    if _has_code(text, 404) or any(word in text for word in (
            'not_found', 'not found', 'model_not_found', 'decommissioned', 'does not exist')):
        return 'the model name was not found for this key'
    if is_malformed_tool_call(exc):
        return 'the model returned an unusable answer format (temporary glitch)'
    if _has_code(text, 502, 503, 504) or any(word in text for word in (
            'timeout', 'timed out', 'deadline', 'connection', 'connecterror', 'network',
            'name resolution', 'unreachable', 'unavailable', 'high demand', 'overloaded')):
        return 'the service is busy or unreachable right now (temporary outage, timeout or network problem)'
    return 'an unexpected error occurred (details are in the terminal window)'


def check_provider(provider: Provider, timeout: int = 15) -> tuple[bool, str]:
    '''Free check: is the key accepted and does the model exist? Returns (works, short reason).

    Only account metadata is requested, never a generation, because free tiers allow just a few
    generation requests per day (Gemini: about 20 per model) and a status light must not eat them.
    Quota and overload problems show up when a question is asked, and the fallback chain handles them.
    '''
    try:
        if provider.name.startswith('gemini'):
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=provider.api_key, http_options=types.HttpOptions(timeout=timeout * 1000))
            client.models.get(model=provider.model)
        else:
            from groq import Groq
            available = {model.id for model in Groq(api_key=provider.api_key, timeout=timeout,
                                                    max_retries=0).models.list().data}
            if provider.model not in available:  # e.g. a model Groq has retired
                return False, 'the model name was not found for this key'
        return True, 'key accepted'
    except Exception as exc:  # noqa: BLE001 - any failure means "not usable right now"
        log.warning('%s check failed: %s: %s', provider.label, type(exc).__name__,
                    redact(str(exc)[:300], [provider.api_key]))
        return False, describe_error(exc)
