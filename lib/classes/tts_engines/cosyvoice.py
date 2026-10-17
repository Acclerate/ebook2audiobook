from lib.classes.tts_engines.common.headers import *
from lib.classes.tts_engines.common.preset_loader import load_engine_presets

# vendored CosyVoice repository: inference code plus the matcha dependency it
# ships in third_party/. It is not a pip package, so both roots must be put on
# sys.path before importing (matcha first so the vendored copy wins).
_COSYVOICE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ext', 'py', 'CosyVoice'))
_MATCHA_ROOT = os.path.join(_COSYVOICE_ROOT, 'third_party', 'Matcha-TTS')

def _import_cosyvoice()->tuple:
    for path in (_MATCHA_ROOT, _COSYVOICE_ROOT):
        if not os.path.isdir(path):
            error = f'CosyVoice repository not found at {path}'
            raise FileNotFoundError(error)
        if path not in sys.path:
            sys.path.insert(0, path)
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav
    return CosyVoice2, load_wav

class CosyVoice(TTSUtils, TTSRegistry, name='cosyvoice'):

    # transcript of the vendored zero_shot_prompt.wav used as default voice
    DEFAULT_PROMPT_TEXT = '希望你以后能够做的比我还好呦。'

    def __init__(self, session:DictProxy):
        try:
            self.session = session
            self.cache_dir = tts_dir
            self.speakers_path = None
            self.speaker = None
            self.tts_key = self.session['model_cache']
            self.pth_voice_file = None
            self.resampler_cache = {}
            self.resampled_wav_cache = {}
            self.audio_segments = []
            self.models = load_engine_presets(self.session['tts_engine'])
            self.params = {}
            # effective language for TTS (target when translating, else source)
            self.language = self.session.get('language')
            self.language_iso1 = self.session.get('language_iso1')
            if self.session.get('translate_enabled'):
                if self.session.get('translate'):
                    self.language = self.session['translate']
                if self.session.get('translate_iso1'):
                    self.language_iso1 = self.session['translate_iso1']
            fine_tuned = self.session.get('fine_tuned')
            if fine_tuned not in self.models:
                error = f'Invalid fine_tuned model {fine_tuned}. Available models: {list(self.models.keys())}'
                raise ValueError(error)
            model_cfg = self.models[fine_tuned]
            for required_key in ('repo', 'samplerate', 'voice'):
                if required_key not in model_cfg:
                    error = f'fine_tuned model {fine_tuned} is missing required key {required_key}.'
                    raise ValueError(error)
            self.params['samplerate'] = model_cfg['samplerate']
            self.model_repo = model_cfg['repo']
            self.default_voice = model_cfg['voice']
            enough_vram = self.session['free_vram_gb'] > 4.0
            seed = 0
            #random.seed(seed)
            self.amp_dtype = self._apply_gpu_policy(enough_vram=enough_vram, seed=seed)
            self.device = devices['CUDA']['proc'] if self.session['device'] in [devices['CUDA']['proc'], devices['ROCM']['proc'], devices['JETSON']['proc']] else self.session['device']
            # reuse the XTTS speed slider as the synthesis speed factor
            speed = float(self.session.get('xtts_speed', 1.0) or 1.0)
            self.fine_tuned_params = {'speed': max(0.5, min(2.0, speed))}
            self.engine = self.load_engine()
        except Exception as e:
            error = f'__init__() error: {e}'
            raise ValueError(error)

    def load_engine(self)->Any:
        try:
            msg = f'Loading TTS {self.tts_key} model, it takes a while, please be patient…'
            print(msg)
            self.cleanup_memory()
            engine = loaded_tts.get(self.tts_key)
            if not engine:
                from modelscope import snapshot_download
                msg = f'Downloading {self.model_repo} from ModelScope (first run only)…'
                print(msg)
                model_dir = snapshot_download(
                    self.model_repo,
                    local_dir = os.path.join(tts_dir, 'cosyvoice', self.model_repo.rsplit('/', 1)[-1])
                )
                CosyVoice2, _ = _import_cosyvoice()
                # CosyVoice2 moves itself to CUDA when available; the bf16 llm
                # weights only run under the fp16 autocast path, so fp16 must
                # follow cuda availability rather than the session device string
                import torch
                use_fp16 = torch.cuda.is_available()
                engine = CosyVoice2(model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=use_fp16)
                if engine.sample_rate != self.params['samplerate']:
                    self.params['samplerate'] = engine.sample_rate
                from lib.classes.vram_detector import VRAMDetector
                vram_dict = VRAMDetector().detect_vram(self.session['device'], self.session['script_mode'])
                self.session['free_vram_gb'] = vram_dict.get('free_vram_gb', 0)
                if self.session['free_vram_gb'] > self._loaded_tts_size_gb(loaded_tts):
                    loaded_tts[self.tts_key] = engine
            if engine:
                msg = f'TTS {self.tts_key} Loaded!'
                print(msg)
                return engine
            error = 'load_engine(): engine is None'
            raise RuntimeError(error)
        except Exception as e:
            error = f'load_engine() error: {e}'
            raise RuntimeError(error) from e

    def _set_voice(self, voice:str|None)->tuple:
        # custom clone voices are used as-is: CosyVoice does zero-shot cloning
        # from the raw prompt wav, no builtin-speaker machinery involved
        current_voice = voice if voice is not None else self.default_voice
        if not current_voice or not os.path.exists(current_voice):
            error = f'_set_voice() error: voice file {current_voice} does not exist'
            return None, error
        return current_voice, None

    def convert(self, sentence_file:str, sentence:str, **kwargs)->tuple:
        try:
            import torch
            from lib.classes.tts_engines.common.audio import trim_audio, is_audio_data_valid
            if self.engine:
                sentence_parts = self._split_sentence_on_sml(sentence)
                self.params['block_voice'] = kwargs.get('block_voice', self.session['voice'])
                if self.params.get('inline_voice'):
                    self.params['current_voice'] = self.params['inline_voice']
                else:
                    self.params['current_voice'], error = self._set_voice(self.params['block_voice'])
                    if self.params['current_voice'] is None and error is not None:
                        return False, error
                # the frontend takes a wav path and resamples to 16k itself
                prompt_wav = self.params['current_voice']
                # the default prompt has a known transcript, so zero-shot mode is
                # possible; custom clone voices have none and go cross-lingual
                is_default_voice = os.path.abspath(self.params['current_voice']) == os.path.abspath(self.default_voice)
                # cross-lingual mode needs an explicit language tag for non-Chinese text
                text_prefix = '' if self.language_iso1 in (None, '', 'zh') else f'<|{self.language_iso1}|>'
                self.audio_segments = []
                for part in sentence_parts:
                    part = part.strip()
                    if not part:
                        continue
                    if SML_TAG_PATTERN.fullmatch(part):
                        success, error = self._convert_sml(part)
                        if not success:
                            return False, error
                        continue
                    if not any(c.isalnum() for c in part):
                        continue
                    else:
                        trim_audio_buffer = 0.006
                        if part.endswith("'"):
                            part = part[:-1]
                        try:
                            part_tensor = None
                            with torch.inference_mode():
                                if is_default_voice:
                                    inference = self.engine.inference_zero_shot(
                                        part,
                                        self.DEFAULT_PROMPT_TEXT,
                                        prompt_wav,
                                        stream = False,
                                        **self.fine_tuned_params
                                    )
                                else:
                                    inference = self.engine.inference_cross_lingual(
                                        f'{text_prefix}{part}',
                                        prompt_wav,
                                        stream = False,
                                        **self.fine_tuned_params
                                    )
                                # text_normalize may split the part: every chunk
                                # comes back at self.params['samplerate'] as [1, N]
                                chunks = []
                                for model_output in inference:
                                    audio_part = model_output.get('tts_speech')
                                    if audio_part is None:
                                        continue
                                    if torch.is_tensor(audio_part):
                                        audio_part = audio_part.detach().cpu()
                                    if not is_audio_data_valid(audio_part):
                                        error = 'tts_speech not valid'
                                        return False, error
                                    chunks.append(self._tensor_type(audio_part))
                                if not chunks:
                                    error = 'inference produced no audio'
                                    return False, error
                                part_tensor = torch.cat(chunks, dim=1)
                            if part_tensor.numel() == 0:
                                error = 'part_tensor not valid'
                                return False, error
                            if part[-1].isalnum() or part[-1] == '—':
                                part_tensor = trim_audio(part_tensor.squeeze(0), self.params['samplerate'], 0.001, trim_audio_buffer).unsqueeze(0)
                            self.audio_segments.append(part_tensor)
                        except IndexError as e:
                            self.cleanup_memory()
                            error = f'convert() error at {e} segment: {part}'
                            return False, error
                        except Exception as e:
                            # free the failed part's tensors before returning False;
                            # core.py then unloads the engine via unload_tts_manager().
                            self.cleanup_memory()
                            return False, self.log_exception(f'{self.__class__.__name__}.convert() part loop', e)
                if self.audio_segments:
                    segment_tensor = torch.cat(self.audio_segments, dim=-1)
                    if not self.audio_save(sentence_file, segment_tensor, self.params['samplerate']):
                        error = f'audio_save() error: cannot save {sentence_file}'
                        return False, error
                    self.audio_segments = []
                    if not os.path.exists(sentence_file):
                        error = f'Cannot create {sentence_file}'
                        return False, error
                return True, None
            else:
                error = f"TTS engine {self.session['tts_engine']} failed to load!"
                return False, error
        except Exception as e:
            self.cleanup_memory()
            self.audio_segments = []
            return False, self.log_exception(f'{self.__class__.__name__}.convert()',e)

    def create_vtt(self, all_sentences:list)->bool:
        if self._build_vtt_file(all_sentences):
            return True
        return False
