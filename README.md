# Twitch Chess Highlight Generator

Local, deterministic Python CLI for generating highlight clips from Twitch VOD
files. It is tuned for chess streams where emotional spikes from the streamer
are often visible as audio RMS peaks.

## Requirements

- Python 3.10+
- FFmpeg and FFprobe on `PATH`
- No Python package dependencies are required, but this command is harmless:

```powershell
pip install -r requirements.txt
```

## Usage

```powershell
python highlight_generator.py input.mp4
```

Outputs clips to `highlights/highlight_001.mp4`, `highlight_002.mp4`, and so on,
plus `highlights/highlights.json`.

Useful options:

```powershell
python highlight_generator.py input.mp4 --dry-run
python highlight_generator.py input.mp4 -o output_clips
python highlight_generator.py input.mp4 --reencode
python highlight_generator.py input.mp4 --config config.json
```

`--reencode` is slower but gives more accurate clip boundaries than FFmpeg
stream copy.

## Config Example

JSON keys match the `Config` fields in `highlight_generator.py`.

```json
{
  "audio_threshold": 0.07,
  "min_peak_distance": 8,
  "pre_roll_seconds": 15,
  "post_roll_seconds": 35,
  "merge_within_seconds": 50
}
```

The uppercase constants at the top of `highlight_generator.py` are the primary
defaults requested by the spec:

- `AUDIO_THRESHOLD`
- `MIN_PEAK_DISTANCE`
- `PRE_ROLL_SECONDS`
- `POST_ROLL_SECONDS`

## How It Works

1. FFmpeg streams mono PCM audio from the video.
2. Python computes short-time RMS energy in small frames using streaming
   standard-library signal processing.
3. The detector finds local RMS peaks above a threshold.
4. Nearby peaks are merged into highlight windows.
5. Each highlight receives pre-roll, post-roll, and a confidence score.
6. FFmpeg exports each clip and writes structured JSON metadata.

The code includes a local silence-gap burst signal: if a spike follows several
quiet seconds, it receives a small confidence bonus.
