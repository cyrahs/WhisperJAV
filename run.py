#!/usr/bin/env python3
"""
WhisperJAV Pipeline Runner

A simple pipeline script that:
1. Creates a temp directory
2. Uses ffmpeg to extract WAV audio from video
3. Uses whisperjav to extract subtitles (two-pass ensemble)
4. Translates with whisperjav-translate
5. Copies subtitles to the output directory
6. Copies the final Chinese subtitle to the media directory

Configuration is loaded from environment variables (recommended: `output/.env`,
relative to this script). Values in the real environment take precedence over
`.env`.

Required env:
    SOURCE_LANGUAGE
    TRANSLATION_TARGET
    TRANSLATION_PROVIDER
    TRANSLATION_MODEL
    TRANSLATION_TONE

Optional env:
    TRANSLATION_API_KEY (can also use provider-specific env var)
    TRANSLATION_ENDPOINT (custom API endpoint for OpenAI-compatible APIs)
    MEDIA_DIR_MAPPINGS (for media copy step)

Usage:
    python run.py /path/to/video.mp4
    python run.py /path/to/video.mp4 --skip-translation
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional


class Tee:
    """Writes to multiple streams simultaneously."""
    
    def __init__(self, *streams):
        self.streams = streams
    
    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
    
    def flush(self):
        for stream in self.streams:
            stream.flush()
    
    def fileno(self):
        # Return the fileno of the first stream (usually stdout)
        return self.streams[0].fileno()

# Use virtual environment Python if available
VENV_PYTHON = Path(__file__).parent / ".venv" / "bin" / "python"
PYTHON_EXE = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

# Default output directory
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# =============================================================================
# ENV LOADING
# =============================================================================

def load_env_file(env_path: Path) -> None:
    """Load simple KEY=VALUE lines from a .env file into os.environ.

    Existing environment variables take precedence over file values.
    """
    if not env_path.exists():
        return

    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("export "):
                line = line[len("export "):].strip()

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue

            if len(value) >= 2 and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
                value = value[1:-1]
            else:
                for sep in (" #", "\t#"):
                    if sep in value:
                        value = value.split(sep, 1)[0].rstrip()
                        break

            os.environ.setdefault(key, value)
    except Exception as e:
        print(f"[WARNING] Failed to load env file: {env_path} ({e})", file=sys.stderr)


_DEFAULT_ENV_PATH = Path(__file__).resolve().parent / "output" / ".env"
load_env_file(_DEFAULT_ENV_PATH)

# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================


SOURCE_LANGUAGE = os.getenv("SOURCE_LANGUAGE")
TRANSLATION_TARGET = os.getenv("TRANSLATION_TARGET")
TRANSLATION_PROVIDER = os.getenv("TRANSLATION_PROVIDER")
TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL")
TRANSLATION_TONE = os.getenv("TRANSLATION_TONE")  # "standard" or "pornify"
TRANSLATION_API_KEY = os.getenv("TRANSLATION_API_KEY")
TRANSLATION_ENDPOINT = os.getenv("TRANSLATION_ENDPOINT")  # Custom API endpoint for OpenAI-compatible APIs

# Media directory mapping (used for the final Chinese subtitle copy step).
# Configure via MEDIA_DIR_MAPPINGS in `output/.env` (or real environment).
# Example:
#   MEDIA_DIR_MAPPINGS=/A:/B
#   /A/TEST/TEST-001.mp4 -> /B/TEST/TEST-001/TEST-001.chs.srt
def load_media_dir_mappings(env_value: Optional[str]) -> Optional[list[tuple[Path, Path]]]:
    """Load MEDIA_DIR_MAPPINGS from env.

    Supported formats:
      - JSON: [["/src/root", "/dst/root"], ...]
      - "source:target;source:target" (also supports "=" and "=>")
    """
    if not env_value:
        return None

    env_value = env_value.strip()
    if not env_value:
        return None

    # Prefer JSON for unambiguous parsing
    if env_value.startswith("["):
        try:
            parsed = json.loads(env_value)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in MEDIA_DIR_MAPPINGS: {e}") from e

        if not isinstance(parsed, list):
            raise ValueError("MEDIA_DIR_MAPPINGS JSON must be a list of [source, target] pairs")

        mappings: list[tuple[Path, Path]] = []
        for item in parsed:
            if not (isinstance(item, (list, tuple)) and len(item) == 2):
                raise ValueError("MEDIA_DIR_MAPPINGS entries must be [source, target]")
            source, target = item
            if not isinstance(source, str) or not isinstance(target, str):
                raise ValueError("MEDIA_DIR_MAPPINGS source/target must be strings")
            mappings.append((Path(source), Path(target)))

        return mappings or None

    # Fallback: source:target;source:target (also supports = and =>)
    mappings = []
    for entry in [p.strip() for p in env_value.split(";") if p.strip()]:
        if "=>" in entry:
            source, target = entry.split("=>", 1)
        elif "=" in entry:
            source, target = entry.split("=", 1)
        elif ":" in entry:
            source, target = entry.split(":", 1)
        else:
            raise ValueError("MEDIA_DIR_MAPPINGS must be JSON or 'source:target;source:target'")
        mappings.append((Path(source.strip()), Path(target.strip())))

    return mappings or None


try:
    MEDIA_DIR_MAPPINGS = load_media_dir_mappings(os.getenv("MEDIA_DIR_MAPPINGS")) or []
except Exception as e:
    print(f"[WARNING] Invalid MEDIA_DIR_MAPPINGS, media copy disabled: {e}", file=sys.stderr)
    MEDIA_DIR_MAPPINGS = []


def infer_media_subtitle_dir(input_video: Path) -> Optional[Path]:
    for source_root, target_root in MEDIA_DIR_MAPPINGS:
        try:
            relative_path = input_video.relative_to(source_root)
        except ValueError:
            continue
        return target_root / relative_path.parent / input_video.stem
    return None


def run_command(cmd: list, description: str, capture_output: bool = False) -> Optional[str]:
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"[STEP] {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(str(c) for c in cmd)}\n")
    
    try:
        if capture_output:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            if result.returncode != 0:
                print(f"Error: {result.stderr}", file=sys.stderr)
                return None
            return result.stdout.strip()
        else:
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"Command failed with exit code {result.returncode}", file=sys.stderr)
                return None
            return "success"
    except FileNotFoundError as e:
        print(f"Error: Command not found - {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error running command: {e}", file=sys.stderr)
        return None


def extract_audio(input_video: Path, output_wav: Path) -> bool:
    """Extract WAV audio from video using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-i", str(input_video),
        "-vn",                    # No video
        "-acodec", "pcm_s16le",   # 16-bit PCM WAV
        "-ar", "48000",           # 48kHz sample rate (aligned with whisperjav)
        "-ac", "1",               # Mono
        "-y",                     # Overwrite output
        str(output_wav)
    ]
    return run_command(cmd, f"Extracting audio from {input_video.name}") is not None


def run_whisperjav(audio_file: Path, output_dir: Path, temp_dir: Path) -> Optional[Path]:
    """Run whisperjav with two-pass ensemble pipeline.
    
    Uses configuration constants:
        - SOURCE_LANGUAGE: Source audio language
    
    Ensemble Configuration:
        Pass 1: Fidelity | Balanced | TEN | Large-V2
        Pass 2: Balanced | Aggressive | Silero V4 | Large-V2
        Merge: pass1_primary (pass 1 + gap filled with pass 2)
    """
    cmd = [
        PYTHON_EXE, "-m", "whisperjav.main",
        str(audio_file),
        "--output-dir", str(output_dir),
        "--temp-dir", str(temp_dir),
        "--language", SOURCE_LANGUAGE,
        "--subs-language", "native",
        # Two-pass ensemble mode
        "--ensemble",
        # Pass 1: Fidelity | Balanced sensitivity | TEN segmenter | Large-V2
        "--pass1-pipeline", "fidelity",
        "--pass1-sensitivity", "balanced",
        "--pass1-speech-segmenter", "ten",
        "--pass1-model", "large-v2",
        # Pass 2: Balanced | Aggressive sensitivity | Silero V4 | Large-V2
        "--pass2-pipeline", "balanced",
        "--pass2-sensitivity", "aggressive",
        "--pass2-speech-segmenter", "silero",
        "--pass2-model", "large-v2",
        # Merge strategy: pass 1 primary + gap fill with pass 2
        "--merge-strategy", "pass1_primary",
    ]
    
    result = run_command(cmd, f"Transcribing {audio_file.name} with WhisperJAV (Two-Pass Ensemble)")
    if result is None:
        return None
    
    # Find the generated subtitle file
    lang_code = {"japanese": "ja", "korean": "ko", "chinese": "zh", "english": "en"}.get(SOURCE_LANGUAGE, "ja")
    expected_srt = output_dir / f"{audio_file.stem}.{lang_code}.whisperjav.srt"
    
    if expected_srt.exists():
        return expected_srt
    
    # Fallback: find any .srt file in output dir
    srt_files = list(output_dir.glob("*.srt"))
    if srt_files:
        return srt_files[0]
    
    print(f"Error: No subtitle file found in {output_dir}", file=sys.stderr)
    return None


def run_translation(input_srt: Path) -> Optional[Path]:
    """Run whisperjav-translate to translate subtitles.
    
    Uses configuration constants:
        - TRANSLATION_TARGET: Target language
        - TRANSLATION_PROVIDER: AI provider (deepseek, openrouter, gemini, etc.)
        - TRANSLATION_MODEL: Model name for the provider
        - TRANSLATION_TONE: Translation style (standard, pornify)
        - TRANSLATION_API_KEY: API key (or uses environment variable if empty)
        - TRANSLATION_ENDPOINT: Custom API endpoint for OpenAI-compatible APIs
    """
    cmd = [
        PYTHON_EXE, "-m", "whisperjav.translate.cli",
        "-i", str(input_srt),
        "--source", SOURCE_LANGUAGE,
        "--target", TRANSLATION_TARGET,
        "--provider", TRANSLATION_PROVIDER,
        "--model", TRANSLATION_MODEL,
        "--tone", TRANSLATION_TONE,
        "--stream"
    ]
    
    if TRANSLATION_API_KEY:
        cmd.extend(["--api-key", TRANSLATION_API_KEY])
    
    if TRANSLATION_ENDPOINT:
        cmd.extend(["--endpoint", TRANSLATION_ENDPOINT])
    
    # Run translation, capturing stdout for output path
    print(f"\n{'='*60}")
    print(f"[STEP] Translating to {TRANSLATION_TARGET} using {TRANSLATION_PROVIDER}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(str(c) for c in cmd)}\n")
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=None,  # Let stderr flow to console
            text=True,
            encoding='utf-8'
        )
        
        if result.returncode != 0:
            print(f"Translation failed with exit code {result.returncode}", file=sys.stderr)
            return None
        
        # stdout contains the output path
        output_path = result.stdout.strip()
        if output_path and Path(output_path).exists():
            return Path(output_path)
        
        # Fallback: look for translated file
        stem = input_srt.stem
        if stem.endswith('.whisperjav'):
            stem = stem[:-11]  # Remove .whisperjav suffix
        expected = input_srt.parent / f"{stem}.{TRANSLATION_TARGET}.srt"
        if expected.exists():
            return expected
            
    except Exception as e:
        print(f"Error running translation: {e}", file=sys.stderr)
    
    return None


def copy_subtitles(srt_files: list, destination_dir: Path, video_stem: str) -> list:
    """Copy subtitle files to destination directory with proper naming."""
    # Normalize language names to short codes
    lang_normalize = {
        'japanese': 'ja',
        'english': 'en',
        'chinese': 'chs',
        'korean': 'ko',
        'zh': 'chs',
    }
    lang_indicators = ['ja', 'en', 'zh', 'ko', 'chs', 'japanese', 'english', 'chinese', 'korean']
    
    copied = []
    for srt in srt_files:
        if srt and srt.exists():
            # Generate destination filename based on video name
            name_parts = srt.stem.split('.')
            
            # Extract language indicator from filename (use LAST match for translated files)
            lang_part = ""
            for part in reversed(name_parts):  # Search from end to find target language
                if part.lower() in lang_indicators:
                    # Normalize to short code
                    normalized = lang_normalize.get(part.lower(), part.lower())
                    lang_part = f".{normalized}"
                    break
            
            dest_name = f"{video_stem}{lang_part}.srt"
            dest_path = destination_dir / dest_name
            
            shutil.copy2(srt, dest_path)
            print(f"Copied: {srt.name} -> {dest_path}")
            copied.append(dest_path)
    
    return copied


def validate_required_env(required: dict[str, Optional[str]]) -> None:
    missing = [name for name, value in required.items() if value is None or str(value).strip() == ""]
    if not missing:
        return

    print("\n[ERROR] Missing required configuration in environment:", file=sys.stderr)
    for name in missing:
        print(f"  - {name}", file=sys.stderr)
    print(f"\nSet them in `{_DEFAULT_ENV_PATH}` (or export them) and try again.", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="WhisperJAV Pipeline - Extract and translate subtitles from video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run.py video.mp4
    python run.py video.mp4 -o /path/to/subs
    python run.py video.mp4 --skip-translation
    python run.py video.mp4 --keep-temp
    
    # Continue from existing temp directory (skip audio extraction)
    python run.py --continue-from /tmp/whisperjav_xxx -o /path/to/subs
    
    # Translate an existing subtitle file
    python run.py --translate /path/to/subtitle.ja.srt

Configuration is loaded from environment variables.

This script loads `output/.env` (relative to `run.py`) if present. Any matching
real environment variables take precedence.
        """
    )
    
    # Input
    parser.add_argument("input", nargs="?", help="Input video file path")
    
    # Continue from existing temp dir
    parser.add_argument("--continue-from", metavar="TEMP_DIR",
                       help="Continue from existing temp directory with extracted audio (skip extraction)")
    
    # Translate-only mode
    parser.add_argument("--translate", metavar="SRT_FILE",
                       help="Translate an existing subtitle file (writes intermediates to temp; copies only final subtitle back)")
    
    # Pipeline options
    parser.add_argument("--skip-translation", action="store_true",
                       help="Skip translation step, only transcribe")
    
    # Output options
    parser.add_argument("-o", "--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                       help=f"Output directory for subtitles (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--keep-temp", action="store_true",
                       help="Keep temporary files after processing")
    parser.add_argument("--temp-dir", help="Custom temporary directory path")
    
    args = parser.parse_args()
    
    # Handle --translate mode (translate-only)
    translate_only_mode = args.translate is not None
    
    if translate_only_mode:
        validate_required_env({
            "SOURCE_LANGUAGE": SOURCE_LANGUAGE,
            "TRANSLATION_TARGET": TRANSLATION_TARGET,
            "TRANSLATION_PROVIDER": TRANSLATION_PROVIDER,
            "TRANSLATION_MODEL": TRANSLATION_MODEL,
            "TRANSLATION_TONE": TRANSLATION_TONE,
        })

        input_srt = Path(args.translate).resolve()
        if not input_srt.exists():
            print(f"Error: Subtitle file not found: {input_srt}", file=sys.stderr)
            sys.exit(1)
        
        if not input_srt.suffix.lower() == '.srt':
            print(f"Error: Expected .srt file, got: {input_srt}", file=sys.stderr)
            sys.exit(1)
        
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create temp directory for translation intermediates so we don't pollute output dir
        temp_base = Path(args.temp_dir) if args.temp_dir else None
        if temp_base:
            temp_base.mkdir(parents=True, exist_ok=True)
        temp_dir = tempfile.mkdtemp(prefix="whisperjav_translate_", dir=str(temp_base) if temp_base else None)
        temp_path = Path(temp_dir)
        
        # Set up logging
        video_stem = input_srt.stem.split('.')[0]  # Get base name (e.g., TEST-001 from TEST-001.ja.srt)
        log_file_path = output_dir / f"{video_stem}.translate.log"
        log_file = open(log_file_path, "w", encoding="utf-8")
        log_file.write(f"WhisperJAV Translation Log - {datetime.now().isoformat()}\n")
        log_file.write(f"{'='*60}\n\n")
        
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║              WhisperJAV Translation Only                     ║
╠══════════════════════════════════════════════════════════════╣
║  Input:    {input_srt.name[:45]:<45}║
║  Target:   {TRANSLATION_TARGET} ({TRANSLATION_TONE})                                   ║
║  Provider: {TRANSLATION_PROVIDER} ({TRANSLATION_MODEL})              ║
║  Output:   {str(output_dir)[:45]:<45}║
║  Temp:     {str(temp_path)[:45]:<45}║
╚══════════════════════════════════════════════════════════════╝
""")
        
        try:
            temp_input_srt = temp_path / input_srt.name
            shutil.copy2(input_srt, temp_input_srt)

            translated_srt = run_translation(temp_input_srt)
            
            if translated_srt:
                # Copy to output directory with proper naming
                final_files = copy_subtitles([translated_srt], output_dir, video_stem)
                
                print(f"\n{'='*60}")
                print("[COMPLETE] Translation finished successfully!")
                print(f"{'='*60}")
                print(f"\nOutput files:")
                for f in final_files:
                    print(f"  - {f}")
            else:
                print("\n[ERROR] Translation failed", file=sys.stderr)
                sys.exit(1)
        finally:
            if not args.keep_temp:
                print(f"\n[CLEANUP] Removing temporary directory: {temp_path}")
                shutil.rmtree(temp_path, ignore_errors=True)
            else:
                print(f"\n[INFO] Temporary files preserved at: {temp_path}")

            if 'log_file' in locals() and log_file:
                print(f"\n[INFO] Log file saved to: {log_file_path}")
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                log_file.close()
        
        sys.exit(0)
    
    # Handle --continue-from mode
    continue_mode = args.continue_from is not None
    
    if continue_mode:
        validate_required_env({"SOURCE_LANGUAGE": SOURCE_LANGUAGE})

        # Continue from existing temp directory
        temp_path = Path(args.continue_from).resolve()
        if not temp_path.exists():
            print(f"Error: Temp directory not found: {temp_path}", file=sys.stderr)
            sys.exit(1)
        
        # Find WAV file in temp directory
        wav_files = list(temp_path.glob("*.wav"))
        if not wav_files:
            print(f"Error: No WAV file found in {temp_path}", file=sys.stderr)
            sys.exit(1)
        
        wav_file = wav_files[0]
        video_stem = wav_file.stem
        
        # Output directory
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        input_display = f"[continue] {wav_file.name}"
    else:
        validate_required_env({"SOURCE_LANGUAGE": SOURCE_LANGUAGE})

        # Normal mode - validate input video
        if not args.input:
            print("Error: Input video file is required (or use --continue-from)", file=sys.stderr)
            sys.exit(1)
        
        input_video = Path(args.input).resolve()
        if not input_video.exists():
            print(f"Error: Input file not found: {input_video}", file=sys.stderr)
            sys.exit(1)
        
        if not input_video.is_file():
            print(f"Error: Input is not a file: {input_video}", file=sys.stderr)
            sys.exit(1)
        
        # Check for ffmpeg
        if shutil.which("ffmpeg") is None:
            print("Error: ffmpeg is not installed or not in PATH", file=sys.stderr)
            sys.exit(1)
        
        video_stem = input_video.stem
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        input_display = input_video.name[:45]
    
    # Set up logging to file
    log_file_path = output_dir / f"{video_stem}.log"
    log_file = open(log_file_path, "w", encoding="utf-8")
    log_file.write(f"WhisperJAV Pipeline Log - {datetime.now().isoformat()}\n")
    log_file.write(f"{'='*60}\n\n")
    
    # Tee stdout and stderr to both console and log file
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)
    
    if not args.skip_translation:
        validate_required_env({
            "TRANSLATION_TARGET": TRANSLATION_TARGET,
            "TRANSLATION_PROVIDER": TRANSLATION_PROVIDER,
            "TRANSLATION_MODEL": TRANSLATION_MODEL,
            "TRANSLATION_TONE": TRANSLATION_TONE,
        })

    target_display = f"{TRANSLATION_TARGET} ({TRANSLATION_TONE})" if not args.skip_translation else "N/A (skip)"
    provider_display = f"{TRANSLATION_PROVIDER} ({TRANSLATION_MODEL})" if not args.skip_translation else "N/A (skip)"
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              WhisperJAV Pipeline Runner                      ║
╠══════════════════════════════════════════════════════════════╣
║  Input:    {input_display:<45}║
║  Source:   {SOURCE_LANGUAGE:<45}║
╠══════════════════════════════════════════════════════════════╣
║  Two-Pass Ensemble Configuration:                            ║
║    Pass 1: Fidelity | Balanced | TEN | Large-V2              ║
║    Pass 2: Balanced | Aggressive | Silero V4 | Large-V2      ║
║    Merge:  pass1_primary (gap-fill with pass 2)              ║
╠══════════════════════════════════════════════════════════════╣
║  Target:   {target_display:<45}║
║  Provider: {provider_display:<45}║
║  Output:   {str(output_dir)[:45]:<45}║
╚══════════════════════════════════════════════════════════════╝
""")
    
    generated_subtitles = []
    
    if continue_mode:
        # Continue mode - use existing temp directory
        print(f"[INFO] Continuing from: {temp_path}")
    else:
        # Create new temp directory
        temp_base = Path(args.temp_dir) if args.temp_dir else None
        temp_dir = tempfile.mkdtemp(prefix="whisperjav_", dir=temp_base)
        temp_path = Path(temp_dir)
        print(f"[INFO] Temporary directory: {temp_path}")
    
    try:
        if continue_mode:
            # Skip audio extraction in continue mode
            print(f"[INFO] Using existing audio: {wav_file}")
        else:
            # Step 1: Extract audio
            wav_file = temp_path / f"{video_stem}.wav"
            if not extract_audio(input_video, wav_file):
                print("Error: Failed to extract audio", file=sys.stderr)
                sys.exit(1)
            
            print(f"[SUCCESS] Audio extracted: {wav_file}")
        
        # Step 2: Run WhisperJAV transcription
        whisperjav_output_dir = temp_path / "subs"
        whisperjav_output_dir.mkdir(exist_ok=True)
        whisperjav_temp_dir = temp_path / "whisperjav_temp"
        whisperjav_temp_dir.mkdir(exist_ok=True)
        
        transcribed_srt = run_whisperjav(
            wav_file, 
            whisperjav_output_dir, 
            whisperjav_temp_dir
        )
        
        if not transcribed_srt:
            print("Error: Failed to generate subtitles", file=sys.stderr)
            sys.exit(1)
        
        print(f"[SUCCESS] Subtitles generated: {transcribed_srt}")
        generated_subtitles.append(transcribed_srt)
        
        # Step 3: Translation (optional)
        if not args.skip_translation:
            translated_srt = run_translation(transcribed_srt)
            
            if translated_srt:
                print(f"[SUCCESS] Translation complete: {translated_srt}")
                generated_subtitles.append(translated_srt)
            else:
                print("[WARNING] Translation failed, but transcription was successful")
        
        # Step 4: Copy subtitles to output directory
        print(f"\n{'='*60}")
        print("[STEP] Copying subtitles to output directory")
        print(f"{'='*60}")
        
        final_files = copy_subtitles(generated_subtitles, output_dir, video_stem)

        # Step 5: Copy final Chinese subtitle to media directory (if applicable)
        media_copy_path = None
        if not continue_mode and not args.skip_translation:
            chinese_output = next((p for p in final_files if p.name.lower().endswith(".chs.srt")), None)
            media_dir = infer_media_subtitle_dir(input_video)
            if chinese_output and media_dir:
                media_dir.mkdir(parents=True, exist_ok=True)
                media_copy_path = media_dir / f"{video_stem}.chs.srt"
                shutil.copy2(chinese_output, media_copy_path)
                print(f"Copied (media): {chinese_output} -> {media_copy_path}")
        
        print(f"\n{'='*60}")
        print("[COMPLETE] Pipeline finished successfully!")
        print(f"{'='*60}")
        print(f"\nOutput files:")
        for f in final_files:
            print(f"  - {f}")
        if media_copy_path:
            print(f"\nMedia subtitle:")
            print(f"  - {media_copy_path}")
        
    finally:
        # Cleanup (don't delete in continue mode unless --keep-temp is explicitly False)
        if continue_mode:
            # In continue mode, always preserve the original temp directory
            print(f"\n[INFO] Temporary files preserved at: {temp_path}")
        elif not args.keep_temp:
            print(f"\n[CLEANUP] Removing temporary directory: {temp_path}")
            shutil.rmtree(temp_path, ignore_errors=True)
        else:
            print(f"\n[INFO] Temporary files preserved at: {temp_path}")
        
        # Close log file and restore stdout/stderr
        if 'log_file' in locals() and log_file:
            print(f"\n[INFO] Log file saved to: {log_file_path}")
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file.close()


if __name__ == "__main__":
    main()
