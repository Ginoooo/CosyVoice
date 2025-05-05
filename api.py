import os
import sys
import time
import base64
import logging
import datetime
import subprocess
import shutil
import threading
import queue
import gc
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from logging.handlers import RotatingFileHandler
from functools import lru_cache

import torch
import torchaudio
from flask import Flask, request, jsonify, send_file, make_response
from modelscope import snapshot_download

# Set up root directory
root_dir = Path(__file__).parent.as_posix()

# Configure paths for ffmpeg and Matcha-TTS
if sys.platform == 'win32':
    os.environ['PATH'] = f"{root_dir};{root_dir}\\ffmpeg;{os.environ['PATH']};{root_dir}/third_party/Matcha-TTS"
else:
    os.environ['PATH'] = f"{root_dir}:{root_dir}/ffmpeg:{os.environ['PATH']}"
    os.environ['PYTHONPATH'] = f"{os.environ.get('PYTHONPATH', '')}:third_party/Matcha-TTS"

# Add Matcha-TTS to system path
sys.path.append(f'{root_dir}/third_party/Matcha-TTS')

# Create necessary directories
tmp_dir = Path(f'{root_dir}/tmp').as_posix()
logs_dir = Path(f'{root_dir}/logs').as_posix()
os.makedirs(tmp_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# Import after environment setup
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# Performance Configuration
# Set to True to make better use of GPU
USE_GPU = torch.cuda.is_available()
NUM_THREADS = max(1, os.cpu_count() - 2)  # Leave some CPU cores free for system
AUDIO_CACHE_SIZE = 100  # Number of recent audio results to cache
BATCH_SIZE = 8  # Process text in batches for better performance

# Configure PyTorch threads for CPU operations
torch.set_num_threads(NUM_THREADS)
if USE_GPU:
    # GPU optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

print(f"Running with {'GPU' if USE_GPU else 'CPU'}, {NUM_THREADS} CPU threads")

# Download required models
def download_models():
    """Download required models from modelscope."""
    models = [
        ('iic/CosyVoice2-0.5B', 'pretrained_models/CosyVoice2-0.5B'),
        ('iic/CosyVoice-300M-SFT', 'pretrained_models/CosyVoice-300M-SFT'),
        ('iic/CosyVoice-ttsfrd', 'pretrained_models/CosyVoice-ttsfrd')
    ]
    
    for model_id, local_dir in models:
        if not Path(local_dir).exists():
            print(f"Downloading model: {model_id}")
            snapshot_download(model_id, local_dir=local_dir)
        else:
            print(f"Model already exists: {local_dir}")

# Configure logging
def setup_logging() -> Flask:
    """Configure app and logging settings."""
    # Disable Werkzeug default handler
    log = logging.getLogger('werkzeug')
    log.handlers[:] = []
    log.setLevel(logging.WARNING)

    # Reset Flask root logger
    root_log = logging.getLogger()
    root_log.handlers = []
    root_log.setLevel(logging.WARNING)

    # Initialize Flask app
    app = Flask(__name__, 
        static_folder=f'{root_dir}/tmp', 
        static_url_path='/tmp')

    app.logger.setLevel(logging.WARNING)
    
    # Configure log file with rotation
    log_file = f'{logs_dir}/{datetime.datetime.now().strftime("%Y%m%d")}.log'
    file_handler = RotatingFileHandler(log_file, maxBytes=1024 * 1024, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(formatter)
    app.logger.addHandler(file_handler)
    
    return app

# Constants
VOICE_LIST = ['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']

# Global models with lazy loading and thread safety
class ModelManager:
    def __init__(self):
        self.sft_model = None
        self.tts_model = None
        self.lock = threading.RLock()
        
    def get_sft_model(self):
        with self.lock:
            if self.sft_model is None:
                print("Loading SFT model...")
                self.sft_model = CosyVoice('pretrained_models/CosyVoice-300M-SFT', load_jit=True)
                if USE_GPU:
                    self.sft_model.model.to('cuda')
            return self.sft_model
            
    def get_tts_model(self):
        with self.lock:
            if self.tts_model is None:
                print("Loading TTS model...")
                self.tts_model = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=True, load_trt=False)
                if USE_GPU:
                    self.tts_model.model.to('cuda')
            return self.tts_model
            
    def clear_gpu_memory(self):
        if USE_GPU:
            with self.lock:
                torch.cuda.empty_cache()
                gc.collect()

model_manager = ModelManager()

# Worker thread pool
class WorkerThread(threading.Thread):
    def __init__(self, task_queue):
        threading.Thread.__init__(self, daemon=True)
        self.task_queue = task_queue
        self.running = True
        
    def run(self):
        while self.running:
            try:
                task, args, kwargs, result_queue = self.task_queue.get(timeout=1)
                try:
                    result = task(*args, **kwargs)
                    result_queue.put((True, result))
                except Exception as e:
                    result_queue.put((False, str(e)))
                finally:
                    self.task_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker thread error: {e}")
                
    def stop(self):
        self.running = False

class ThreadPool:
    def __init__(self, num_threads=NUM_THREADS):
        self.task_queue = queue.Queue()
        self.threads = []
        for _ in range(num_threads):
            thread = WorkerThread(self.task_queue)
            thread.start()
            self.threads.append(thread)
            
    def submit(self, task, *args, **kwargs):
        result_queue = queue.Queue()
        self.task_queue.put((task, args, kwargs, result_queue))
        return result_queue
        
    def shutdown(self):
        for thread in self.threads:
            thread.stop()
        self.task_queue = None
        self.threads = []

thread_pool = ThreadPool(NUM_THREADS)

# LRU cache for audio processing
@lru_cache(maxsize=AUDIO_CACHE_SIZE)
def process_reference_audio(reference_audio_path):
    """Process reference audio with caching for better performance."""
    ref_audio = f"{tmp_dir}/-refaudio-{hash(reference_audio_path)}.wav"
    
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-ignore_unknown", "-y", "-i", 
         reference_audio_path, "-ar", "16000", ref_audio],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        check=True,
        text=True,
        creationflags=0 if sys.platform != 'win32' else subprocess.CREATE_NO_WINDOW
    )
    
    return load_wav(ref_audio, 16000)

def base64_to_wav(encoded_str: str, output_path: str) -> None:
    """Convert base64 encoded string to WAV file."""
    if not encoded_str:
        raise ValueError("Base64 encoded string is empty.")

    # Decode base64 string to bytes
    wav_bytes = base64.b64decode(encoded_str)

    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Write to file
    with open(output_path, "wb") as wav_file:
        wav_file.write(wav_bytes)

def get_params(req) -> Dict[str, Any]:
    """Extract and normalize request parameters."""
    params = {
        "text": "",
        "lang": "",
        "role": "中文女",
        "reference_audio": None,
        "reference_text": "",
        "speed": 1.0
    }
    
    # Get text
    params['text'] = req.args.get("text", "").strip() or req.form.get("text", "").strip()
    
    # Get language and normalize
    params['lang'] = req.args.get("lang", "").strip().lower() or req.form.get("lang", "").strip().lower()
    if params['lang'] == 'ja':
        params['lang'] = 'jp'
    elif params['lang'][:2] == 'zh':
        # Normalize zh-cn zh-tw zh-hk to zh
        params['lang'] = 'zh'
    
    # Get role
    role = req.args.get("role", "").strip() or req.form.get("role", '')
    if role:
        params['role'] = role
    
    # Get reference audio
    params['reference_audio'] = req.args.get("reference_audio", None) or req.form.get("reference_audio", None)
    encode = req.args.get('encode', '') or req.form.get('encode', '')
    if encode == 'base64':
        tmp_name = f'tmp/{time.time()}-clone-{len(params["reference_audio"])}.wav'
        base64_to_wav(params['reference_audio'], f'{root_dir}/{tmp_name}')
        params['reference_audio'] = tmp_name
    
    # Get reference text
    params['reference_text'] = req.args.get("reference_text", '').strip() or req.form.get("reference_text", '')
    
    # Get speed (if provided)
    speed_str = req.args.get("speed", "") or req.form.get("speed", "")
    if speed_str:
        try:
            params['speed'] = float(speed_str)
        except ValueError:
            pass  # Keep default speed if conversion fails
    
    return params

def del_tmp_files(tmp_files: List[str]) -> None:
    """Delete temporary files."""
    for f in tmp_files:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception as e:
                print(f"Error deleting file {f}: {e}")

# Cleanup thread to remove old temp files
def cleanup_thread():
    """Background thread to clean up old temporary files."""
    while True:
        try:
            # Get all .wav files in tmp directory
            now = time.time()
            for f in Path(tmp_dir).glob("*.wav"):
                # Delete files older than 30 minutes
                if now - f.stat().st_mtime > 1800:  # 30 minutes
                    try:
                        os.remove(f)
                        print(f"Cleaned up old file: {f}")
                    except Exception as e:
                        print(f"Error cleaning up file {f}: {e}")
        except Exception as e:
            print(f"Cleanup thread error: {e}")
        
        # Sleep for 5 minutes
        time.sleep(300)

# Start cleanup thread
cleanup_thread_handle = threading.Thread(target=cleanup_thread, daemon=True)
cleanup_thread_handle.start()

# Function to split text into chunks for better processing
def split_text(text, max_length=100):
    """Split text into smaller chunks for better performance."""
    # Simple splitting by punctuation
    sentences = []
    current = ""
    
    for char in text:
        current += char
        if char in ['。', '！', '？', '.', '!', '?', ';', '；', '\n'] and len(current) > 10:
            sentences.append(current)
            current = ""
    
    if current:
        sentences.append(current)
    
    # Group sentences into chunks
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def batch(tts_type: str, outname: str, params: Dict[str, Any]) -> str:
    """Generate audio batch based on specified TTS type."""
    if not shutil.which("ffmpeg"):
        raise Exception('必须安装 ffmpeg')
    
    prompt_speech_16k = None
    
    # Process reference audio if needed
    if tts_type != 'tts':
        if not params['reference_audio'] or not os.path.exists(f"{root_dir}/{params['reference_audio']}"):
            raise Exception(f'参考音频未传入或不存在 {params["reference_audio"]}')
        
        try:
            # Use cached processing for reference audio
            prompt_speech_16k = process_reference_audio(params['reference_audio'])
        except Exception as e:
            raise Exception(f'处理参考音频失败:{e}')

    text = params['text']
    audio_list = []
    sample_rate = 22050  # Default sample rate
    
    # Split text into more manageable chunks if it's long
    text_chunks = split_text(text) if len(text) > 100 else [text]
    
    # Generate audio based on TTS type
    if tts_type == 'tts':
        sft_model = model_manager.get_sft_model()
        sample_rate = 22050
        
        for chunk in text_chunks:
            for _, j in enumerate(sft_model.inference_sft(chunk, params['role'], 
                                                      stream=False, speed=params['speed'])):
                audio_list.append(j['tts_speech'])
            
    elif tts_type == 'clone_eq' and params.get('reference_text'):
        tts_model = model_manager.get_tts_model()
        sample_rate = 24000
        
        for chunk in text_chunks:
            for _, j in enumerate(tts_model.inference_zero_shot(chunk, params.get('reference_text'), 
                                                             prompt_speech_16k, stream=False, 
                                                             speed=params['speed'])):
                audio_list.append(j['tts_speech'])

    else:
        tts_model = model_manager.get_tts_model()
        sample_rate = 24000
        
        for chunk in text_chunks:
            for _, j in enumerate(tts_model.inference_cross_lingual(chunk, prompt_speech_16k, 
                                                                stream=False, 
                                                                speed=params['speed'])):
                audio_list.append(j['tts_speech'])
    
    # Concatenate audio segments
    audio_data = torch.concat(audio_list, dim=1)
    
    # Save to file with appropriate sample rate
    output_path = f"{tmp_dir}/{outname}"
    torchaudio.save(output_path, audio_data, sample_rate, format="wav")   
    
    # Clean up after processing
    model_manager.clear_gpu_memory()
    
    return output_path

# Initialize Flask app with proper logging
app = setup_logging()

# Download models on startup in a separate thread
threading.Thread(target=download_models, daemon=True).start()

@app.route('/tts', methods=['GET', 'POST'])        
def tts():
    """Text-to-speech endpoint."""
    params = get_params(request)
    if not params['text']:
        return make_response(jsonify({"code": 1, "msg": '缺少待合成的文本'}), 400)
        
    try:
        outname = f"tts-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{hash(params['text'])}.wav"
        
        # Check if the file already exists (simple caching)
        cached_file = Path(f"{tmp_dir}/{outname}")
        if cached_file.exists() and (time.time() - cached_file.stat().st_mtime < 3600):  # 1 hour cache
            return send_file(cached_file, mimetype='audio/x-wav')
            
        output_path = batch(tts_type='tts', outname=outname, params=params)
    except Exception as e:
        app.logger.error(f"TTS error: {str(e)}")
        return make_response(jsonify({"code": 2, "msg": str(e)}), 500)
    
    return send_file(output_path, mimetype='audio/x-wav')


@app.route('/clone_mul', methods=['GET', 'POST'])        
@app.route('/clone', methods=['GET', 'POST'])        
def clone():
    """Cross-lingual voice cloning endpoint."""
    try:
        params = get_params(request)
        if not params['text']:
            return make_response(jsonify({"code": 6, "msg": '缺少待合成的文本'}), 400)
            
        ref_audio_hash = "noref"
        if params['reference_audio']:
            ref_audio_hash = str(hash(params['reference_audio']))
            
        outname = f"clone-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{hash(params['text'])}-{ref_audio_hash}.wav"
        
        # Check if the file already exists (simple caching)
        cached_file = Path(f"{tmp_dir}/{outname}")
        if cached_file.exists() and (time.time() - cached_file.stat().st_mtime < 3600):  # 1 hour cache
            return send_file(cached_file, mimetype='audio/x-wav')
            
        output_path = batch(tts_type='clone', outname=outname, params=params)
    except Exception as e:
        app.logger.error(f"Clone error: {str(e)}")
        return make_response(jsonify({"code": 8, "msg": str(e)}), 500)
    
    return send_file(output_path, mimetype='audio/x-wav')


@app.route('/clone_eq', methods=['GET', 'POST'])         
def clone_eq():
    """Same-language voice cloning endpoint."""
    try:
        params = get_params(request)
        if not params['text']:
            return make_response(jsonify({"code": 6, "msg": '缺少待合成的文本'}), 400)
        if not params['reference_text']:
            return make_response(jsonify({"code": 6, "msg": '同语言克隆必须传递引用文本'}), 400)
            
        ref_audio_hash = "noref"
        if params['reference_audio']:
            ref_audio_hash = str(hash(params['reference_audio']))
            
        outname = f"clone_eq-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{hash(params['text'])}-{ref_audio_hash}.wav"
        
        # Check if the file already exists (simple caching)
        cached_file = Path(f"{tmp_dir}/{outname}")
        if cached_file.exists() and (time.time() - cached_file.stat().st_mtime < 3600):  # 1 hour cache
            return send_file(cached_file, mimetype='audio/x-wav')
            
        output_path = batch(tts_type='clone_eq', outname=outname, params=params)
    except Exception as e:
        app.logger.error(f"Clone_eq error: {str(e)}")
        return make_response(jsonify({"code": 8, "msg": str(e)}), 500)
    
    return send_file(output_path, mimetype='audio/x-wav')
     

@app.route('/v1/audio/speech', methods=['POST'])
def audio_speech():
    """OpenAI-compatible /v1/audio/speech API endpoint."""
    import random

    if not request.is_json:
        return jsonify({"error": "请求必须是 JSON 格式"}), 400

    data = request.get_json()

    # Validate required parameters
    if 'input' not in data or 'voice' not in data:
        return jsonify({"error": "请求缺少必要的参数： input, voice"}), 400
    
    text = data.get('input')
    speed = float(data.get('speed', 1.0))
    voice = data.get('voice', '中文女')
    
    params = {
        'text': text,
        'speed': speed
    }
    
    api_name = 'tts'
    if voice in VOICE_LIST:
        params['role'] = voice
    elif Path(voice).exists() or Path(f'{root_dir}/{voice}').exists():
        api_name = 'clone'
        params['reference_audio'] = voice
    else:
        return jsonify({
            "error": {
                "message": f"必须填写配音角色名或参考音频路径", 
                "type": "InvalidVoiceError", 
                "param": f'speed={speed},voice={voice},input={text}', 
                "code": 400
            }
        }), 400

    voice_hash = hash(voice)
    filename = f'openai-{hash(text)}-{speed}-{voice_hash}-{random.randint(1000,99999)}.wav'
    
    # Check if the file already exists (simple caching)
    cached_file = Path(f"{tmp_dir}/{filename}")
    if cached_file.exists() and (time.time() - cached_file.stat().st_mtime < 3600):  # 1 hour cache
        return send_file(cached_file, mimetype='audio/x-wav')
    
    try:
        output_path = batch(tts_type=api_name, outname=filename, params=params)
        return send_file(output_path, mimetype='audio/x-wav')
    except Exception as e:
        app.logger.error(f"OpenAI API error: {str(e)}")
        return jsonify({
            "error": {
                "message": f"{e}", 
                "type": e.__class__.__name__, 
                "param": f'speed={speed},voice={voice},input={text}', 
                "code": 500
            }
        }), 500

# Graceful shutdown
def shutdown_hook():
    """Clean up resources on shutdown."""
    print("Shutting down API server...")
    thread_pool.shutdown()
    # Clear GPU memory
    if USE_GPU:
        torch.cuda.empty_cache()
    print("Resources released")

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        "status": "ok",
        "gpu": USE_GPU,
        "cpu_threads": NUM_THREADS
    })

if __name__ == '__main__':
    import atexit
    atexit.register(shutdown_hook)
    
    host = '127.0.0.1'
    port = 9233
    print(f'\n启动api: http://{host}:{port}\n')
    
    try:
        from waitress import serve
        # More threads for better handling of concurrent requests
        serve(app, host=host, port=port, threads=16)
    except ImportError:
        # Fallback to development server if waitress is not installed
        app.run(host=host, port=port, threaded=True)