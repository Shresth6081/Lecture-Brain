import whisper

from transcriber.whisper_engine import get_model

# ISO-639-1 → human-readable name for the most common lecture languages (get the top 30 supported languages from web)
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",   "hi": "Hindi",      "zh": "Chinese",
    "es": "Spanish",   "fr": "French",     "de": "German",
    "ja": "Japanese",  "ko": "Korean",     "ar": "Arabic",
    "pt": "Portuguese","ru": "Russian",    "it": "Italian",
    "nl": "Dutch",     "pl": "Polish",     "tr": "Turkish",
    "sv": "Swedish",   "da": "Danish",     "fi": "Finnish",
    "no": "Norwegian", "cs": "Czech",      "uk": "Ukrainian",
    "vi": "Vietnamese","th": "Thai",       "id": "Indonesian",
    "ms": "Malay",     "ro": "Romanian",   "hu": "Hungarian",
}


def detect_language(audio_path: str) -> dict:
    model = get_model()

    # Whisper's helper pads / trims to exactly 30 s of audio
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)

    lang_code   = max(probs, key=probs.get)
    confidence  = round(probs[lang_code] * 100, 1)

    return {
        "code": lang_code,
        "name": LANGUAGE_NAMES.get(lang_code, lang_code.upper()),
        "confidence": confidence,
    }