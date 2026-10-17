from lib.conf_models import TTS_ENGINES, default_engine_settings

models = {
    "internal": {
        "lang": "multi",
        "repo": "iic/CosyVoice2-0.5B",
        "voice": default_engine_settings[TTS_ENGINES['COSYVOICE']]['voice'],
        "files": default_engine_settings[TTS_ENGINES['COSYVOICE']]['files'],
        "samplerate": default_engine_settings[TTS_ENGINES['COSYVOICE']]['samplerate']
    }
}
