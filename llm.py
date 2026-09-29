'''LLM providers: Gemini (primary), Groq (automatic fallback), and friendly error messages.'''
from __future__ import annotations
import logging
import os
from dataclasses import dataclass, field
from langchain_core.language_models import BaseChatModel

log = logging.getLogger('docassistant')
# The Google SDK prints a harmless "automatic function calling" notice on every request.
for _name in ('google_genai', 'google_genai.models', 'google.genai'):
    logging.getLogger(_name).setLevel(logging.ERROR)

# Single source of truth for model names. Retired Gemini models return 404 NOT_FOUND
# for new API keys, so a stale .env value is silently replaced by the default.
DEFAULT_GEMINI_MODEL = 'gemini-3.6-flash'
DEFAULT_GROQ_MODEL = 'llama-3.3-70b-versatile'
# Last-resort model on the same Gemini key, used when the primary model is overloaded.
DEFAULT_GEMINI_BACKUP_MODEL = 'gemini-3.5-flash-lite'
DEPRECATED_GEMINI_MODELS = {
    'gemini-2.5-flash', 'gemini-2.5-pro',
    'gemini-1.5-flash', 'gemini-1.5-flash-001', 'gemini-1.5-pro', 'gemini-pro',
}
_PLACEHOLDER_HINTS = ('your_', 'paste_', 'placeholder', 'changeme', 'xxxx')


def is_real_key(key: str | None) -> bool:
    '''True when a value looks like a real API key rather than empty text or a template.'''
    value = (key or '').strip().lower()
    return bool(value) and not any(hint in value for hint in _PLACEHOLDER_HINTS)


def resolve_gemini_model(model_name: str | None = None) -> str:
    '''Explicit argument > GEMINI_MODEL env > default. Blank or retired names use the default.'''
    candidate = (model_name or os.getenv('GEMINI_MODEL') or '').strip()
    return DEFAULT_GEMINI_MODEL if not candidate or candidate in DEPRECATED_GEMINI_MODELS else candidate


def resolve_gemini_backup_model() -> str:
    '''GEMINI_BACKUP_MODEL env > default.'''
    return (os.getenv('GEMINI_BACKUP_MODEL') or '').strip() or DEFAULT_GEMINI_BACKUP_MODEL


def resolve_groq_model(model_name: str | None = None) -> str:
    '''Explicit argument > GROQ_MODEL env > default.'''
    return (model_name or os.getenv('GROQ_MODEL') or '').strip() or DEFAULT_GROQ_MODEL


@dataclass(frozen=True)
class Provider:
    '''One configured LLM service. The key is never shown in repr() or logs.'''
    name: str     # 'gemini', 'groq' or 'gemini-backup'
    label: str    # human-friendly name for the UI
    model: str
    api_key: str = field(repr=False)


def configured_providers(gemini_key: str | None = None, groq_key: str | None = None) -> list[Provider]:
    '''Providers that have a usable key, in priority order: Gemini, then Groq, then a backup
    Gemini model (same key) as a last resort when Gemini's main model is overloaded.
    Setting PRIMARY_LLM=groq in .env puts Groq first instead.

    Keys passed in (for example from the sidebar) win over the environment / .env file.
    '''
    gemini_key = gemini_key if is_real_key(gemini_key) else os.getenv('GOOGLE_API_KEY')
    groq_key = groq_key if is_real_key(groq_key) else os.getenv('GROQ_API_KEY')
    gemini = Provider('gemini', 'Gemini', resolve_gemini_model(), gemini_key.strip())         if is_real_key(gemini_key) else None
    groq = Provider('groq', 'Groq', resolve_groq_model(), groq_key.strip()) if is_real_key(groq_key) else None
    ordered = [groq, gemini] if (os.getenv('PRIMARY_LLM') or '').strip().lower() == 'groq' else [gemini, groq]
    providers = [provider for provider in ordered if provider]
    if gemini and resolve_gemini_backup_model() != gemini.model:
        providers.append(Provider('gemini-backup', 'Gemini backup', resolve_gemini_backup_model(), gemini.api_key))
    return providers


def build_chat_model(provider: Provider, timeout: int = 45, max_retries: int = 1) -> BaseChatModel:
    '''Create the LangChain chat model for a provider (imports are lazy so unused SDKs never load).'''
    if provider.name.startswith('gemini'):
        from langchain_google_genai import ChatGoogleGenerativeAI
        # No temperature: Gemini 3.x uses fixed sampling defaults and warns if one is passed.
        return ChatGoogleGenerativeAI(
            model=provider.model, google_api_key=provider.api_key,
            max_output_tokens=4096, timeout=timeout, max_retries=max_retries,
        )
    if provider.name == 'groq':
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=provider.model, api_key=provider.api_key, temperature=0.1,
            max_tokens=4096, timeout=timeout, max_retries=max_retries,
        )
    raise ValueError(f'Unknown LLM provider: {provider.name}')


def redact(text: str, secrets: list[str]) -> str:
    '''Remove any API key that an upstream error message might echo back.'''
    for secret in secrets:
        if secret and len(secret) > 6:
            text = text.replace(secret, '***')
    return text


def describe_error(exc: BaseException) -> str:
    '''Turn an upstream exception into a short, safe explanation (never the raw message).'''
    text = f'{type(exc).__name__} {exc}'.lower()
    if any(word in text for word in ('api key not valid', 'api_key_invalid', 'invalid api key',
                                     'invalid_api_key', 'unauthorized', 'permission_denied',
                                     'authentication', '401', '403')):
        return 'the API key was rejected (invalid, expired or not allowed)'
    if any(word in text for word in ('resource_exhausted', 'quota', 'rate limit', 'rate_limit',
                                     'too many requests', '429')):
        return 'the free quota or rate limit has been reached'
    if any(word in text for word in ('not_found', 'not found', '404', 'model_not_found')):
        return 'the model name was not found for this key'
    if any(word in text for word in ('timeout', 'timed out', 'deadline', 'connection', 'connecterror',
                                     'network', 'name resolution', 'unreachable', '503', '502',
                                     '504', 'unavailable', 'high demand', 'overloaded')):
        return 'the service is busy or unreachable right now (temporary outage, timeout or network problem)'
    return 'an unexpected error occurred (details are in the terminal window)'


def is_quota_error(exc: BaseException) -> bool:
    '''True for "quota / rate limit" failures, which usually last much longer than an outage.'''
    text = f'{type(exc).__name__} {exc}'.lower()
    return any(word in text for word in ('resource_exhausted', 'quota', 'rate limit', 'rate_limit',
                                         'too many requests', '429'))


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
            Groq(api_key=provider.api_key, timeout=timeout, max_retries=0).models.list()
        return True, 'key accepted'
    except Exception as exc:  # noqa: BLE001 - any failure means "not usable right now"
        log.warning('%s check failed: %s: %s', provider.label, type(exc).__name__,
                    redact(str(exc)[:300], [provider.api_key]))
        return False, describe_error(exc)
