#!/usr/bin/env python3
"""
Generate highlight clips from Twitch VOD recordings, tuned for chess streams.

The detector is intentionally local and deterministic: FFmpeg streams mono audio
from the video, Python computes short-time RMS energy, peak clusters are merged
into highlight windows, and FFmpeg exports the final clips.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median, pstdev
from typing import Any


# -----------------------------
# Clear configuration constants
# -----------------------------

AUDIO_THRESHOLD = 0.08
MIN_PEAK_DISTANCE = 8.0
PRE_ROLL_SECONDS = 15.0
POST_ROLL_SECONDS = 30.0

# Additional tuning knobs. These can also be overridden by a JSON config file.
ANALYSIS_SAMPLE_RATE = 16_000
FRAME_SECONDS = 0.50
MERGE_WITHIN_SECONDS = 45.0
MIN_HIGHLIGHT_SECONDS = 20.0
MAX_HIGHLIGHT_SECONDS = 120.0
ADAPTIVE_THRESHOLD_STD_MULTIPLIER = 2.25
SILENCE_RMS_THRESHOLD = 0.015
SILENCE_LOOKBACK_SECONDS = 6.0
SILENCE_SPIKE_BONUS = 0.15
FFMPEG_COPY_STREAMS = True


@dataclass(frozen=True)
class Config:
    audio_threshold: float = AUDIO_THRESHOLD
    min_peak_distance: float = MIN_PEAK_DISTANCE
    pre_roll_seconds: float = PRE_ROLL_SECONDS
    post_roll_seconds: float = POST_ROLL_SECONDS
    analysis_sample_rate: int = ANALYSIS_SAMPLE_RATE
    frame_seconds: float = FRAME_SECONDS
    merge_within_seconds: float = MERGE_WITHIN_SECONDS
    min_highlight_seconds: float = MIN_HIGHLIGHT_SECONDS
    max_highlight_seconds: float = MAX_HIGHLIGHT_SECONDS
    adaptive_threshold_std_multiplier: float = ADAPTIVE_THRESHOLD_STD_MULTIPLIER
    silence_rms_threshold: float = SILENCE_RMS_THRESHOLD
    silence_lookback_seconds: float = SILENCE_LOOKBACK_SECONDS
    silence_spike_bonus: float = SILENCE_SPIKE_BONUS
    ffmpeg_copy_streams: bool = FFMPEG_COPY_STREAMS


@dataclass
class Peak:
    time: float
    rms: float
    score: float
    duration: float
    silence_burst: bool = False


@dataclass
class Highlight:
    index: int
    start: float
    end: float
    peak_time: float
    confidence: float
    peak_count: int
    max_rms: float
    output_file: str | None = None


def load_config(path: Path | None) -> Config:
    if path is None:
        return Config()

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    normalized = {key.lower(): value for key, value in raw.items()}
    valid_fields = Config.__dataclass_fields__.keys()
    unknown = sorted(set(normalized) - set(valid_fields))
    if unknown:
        raise ValueError(f"Unknown config field(s): {', '.join(unknown)}")

    return Config(**normalized)


def ensure_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"{name} was not found on PATH. Install FFmpeg and try again.")


def run_command(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Command failed with exit code {exc.returncode}: {' '.join(command)}") from exc


def get_video_duration(input_path: Path) -> float | None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None

    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def stream_rms_frames(input_path: Path, config: Config) -> tuple[list[float], list[float]]:
    """
    Stream decoded audio frames from FFmpeg and return per-frame timestamps/RMS.

    Only a compact RMS timeline is kept in memory. At 0.5s frames, a 10-hour VOD
    produces roughly 72k float values, which is small enough for normal laptops.
    """
    samples_per_frame = max(1, int(config.analysis_sample_rate * config.frame_seconds))
    bytes_per_frame = samples_per_frame * 2

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(config.analysis_sample_rate),
        "-f",
        "s16le",
        "-",
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdout is None:
        raise RuntimeError("Failed to open FFmpeg stdout pipe.")

    rms_values: list[float] = []
    times: list[float] = []
    frame_index = 0

    try:
        while True:
            chunk = process.stdout.read(bytes_per_frame)
            if not chunk:
                break
            if len(chunk) < 2:
                continue

            samples = array("h")
            samples.frombytes(chunk[: len(chunk) - (len(chunk) % 2)])
            if sys.byteorder != "little":
                samples.byteswap()
            if not samples:
                continue

            rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples)) / 32768.0
            midpoint = (frame_index + 0.5) * config.frame_seconds
            rms_values.append(rms)
            times.append(midpoint)
            frame_index += 1
    finally:
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed:\n{stderr.strip()}")

    return times, rms_values


def smooth(values: list[float], window: int = 3) -> list[float]:
    if not values or window <= 1:
        return values
    radius = window // 2
    smoothed: list[float] = []
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        smoothed.append(sum(values[start:end]) / (end - start))
    return smoothed


def compute_detection_threshold(rms_values: list[float], config: Config) -> float:
    if not rms_values:
        return config.audio_threshold

    baseline = float(median(rms_values))
    std = float(pstdev(rms_values)) if len(rms_values) > 1 else 0.0
    adaptive = baseline + (std * config.adaptive_threshold_std_multiplier)
    return max(config.audio_threshold, adaptive)


def is_local_maximum(values: list[float], index: int) -> bool:
    left = values[index - 1] if index > 0 else -math.inf
    right = values[index + 1] if index + 1 < len(values) else -math.inf
    return values[index] >= left and values[index] >= right


def estimate_spike_duration(
    rms_values: list[float],
    peak_index: int,
    threshold: float,
    frame_seconds: float,
) -> float:
    low_threshold = threshold * 0.75
    left = peak_index
    while left > 0 and rms_values[left - 1] >= low_threshold:
        left -= 1

    right = peak_index
    while right + 1 < len(rms_values) and rms_values[right + 1] >= low_threshold:
        right += 1

    return max(frame_seconds, (right - left + 1) * frame_seconds)


def detect_silence_burst(rms_values: list[float], peak_index: int, config: Config) -> bool:
    lookback_frames = max(1, int(config.silence_lookback_seconds / config.frame_seconds))
    start = max(0, peak_index - lookback_frames)
    history = rms_values[start:peak_index]
    if not history:
        return False
    return float(median(history)) <= config.silence_rms_threshold


def detect_peaks(times: list[float], rms_values: list[float], config: Config) -> tuple[list[Peak], float]:
    if not times:
        return [], config.audio_threshold

    smoothed = smooth(rms_values, window=3)
    threshold = compute_detection_threshold(smoothed, config)
    min_distance_frames = max(1, int(config.min_peak_distance / config.frame_seconds))

    candidate_indexes = [
        i for i, value in enumerate(smoothed) if value >= threshold and is_local_maximum(smoothed, i)
    ]
    candidate_indexes.sort(key=lambda i: float(smoothed[i]), reverse=True)

    selected: list[int] = []
    for index in candidate_indexes:
        if all(abs(index - kept) >= min_distance_frames for kept in selected):
            selected.append(index)

    selected.sort()

    peaks: list[Peak] = []
    for index in selected:
        duration = estimate_spike_duration(smoothed, index, threshold, config.frame_seconds)
        magnitude = float(smoothed[index] / max(threshold, 1e-9))
        silence_burst = detect_silence_burst(smoothed, index, config)
        score = min(1.0, (magnitude - 1.0) / 2.0 + min(duration / 8.0, 0.35))
        if silence_burst:
            score = min(1.0, score + config.silence_spike_bonus)

        peaks.append(
            Peak(
                time=float(times[index]),
                rms=float(smoothed[index]),
                score=round(score, 4),
                duration=round(duration, 3),
                silence_burst=silence_burst,
            )
        )

    return peaks, threshold


def build_highlights(peaks: list[Peak], duration: float | None, config: Config) -> list[Highlight]:
    if not peaks:
        return []

    groups: list[list[Peak]] = []
    current: list[Peak] = [peaks[0]]

    for peak in peaks[1:]:
        if peak.time - current[-1].time <= config.merge_within_seconds:
            current.append(peak)
        else:
            groups.append(current)
            current = [peak]
    groups.append(current)

    highlights: list[Highlight] = []
    for group in groups:
        primary_peak = max(group, key=lambda peak: (peak.score, peak.rms))
        start = max(0.0, group[0].time - config.pre_roll_seconds)
        end = group[-1].time + config.post_roll_seconds

        if end - start < config.min_highlight_seconds:
            pad = (config.min_highlight_seconds - (end - start)) / 2.0
            start = max(0.0, start - pad)
            end += pad

        if end - start > config.max_highlight_seconds:
            center = primary_peak.time
            half = config.max_highlight_seconds / 2.0
            start = max(0.0, center - half)
            end = start + config.max_highlight_seconds

        if duration is not None:
            end = min(duration, end)
            if end - start < config.min_highlight_seconds:
                start = max(0.0, end - config.min_highlight_seconds)

        cluster_bonus = min(0.25, 0.05 * (len(group) - 1))
        confidence = min(1.0, primary_peak.score + cluster_bonus)

        highlights.append(
            Highlight(
                index=len(highlights) + 1,
                start=round(start, 3),
                end=round(end, 3),
                peak_time=round(primary_peak.time, 3),
                confidence=round(confidence, 4),
                peak_count=len(group),
                max_rms=round(max(peak.rms for peak in group), 6),
            )
        )

    return highlights


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, seconds)
    whole = int(seconds)
    millis = int(round((seconds - whole) * 1000))
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def export_clip(input_path: Path, output_path: Path, start: float, end: float, config: Config) -> None:
    duration = max(0.0, end - start)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        format_seconds(start),
        "-i",
        str(input_path),
        "-t",
        format_seconds(duration),
    ]

    if config.ffmpeg_copy_streams:
        command.extend(["-c", "copy", "-avoid_negative_ts", "make_zero"])
    else:
        command.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac"])

    command.append(str(output_path))
    run_command(command)


def write_metadata(
    output_dir: Path,
    input_path: Path,
    highlights: list[Highlight],
    config: Config,
    detection_threshold: float,
) -> Path:
    metadata_path = output_dir / f"{input_path.stem}-highlights.json"
    payload: dict[str, Any] = {
        "input_file": str(input_path),
        "detection_threshold": round(detection_threshold, 6),
        "config": asdict(config),
        "highlights": [asdict(highlight) for highlight in highlights],
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return metadata_path


def generate_highlights(
    input_path: Path,
    output_dir: Path,
    config: Config,
    dry_run: bool = False,
) -> tuple[list[Highlight], Path, float]:
    ensure_tool("ffmpeg")
    ensure_tool("ffprobe")

    duration = get_video_duration(input_path)
    times, rms_values = stream_rms_frames(input_path, config)
    peaks, detection_threshold = detect_peaks(times, rms_values, config)
    highlights = build_highlights(peaks, duration, config)

    output_dir.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        for highlight in highlights:
            output_path = output_dir / f"{input_path.stem}-highlight-{highlight.index}.mp4"
            export_clip(input_path, output_path, highlight.start, highlight.end, config)
            highlight.output_file = str(output_path)
    else:
        for highlight in highlights:
            highlight.output_file = str(output_dir / f"{input_path.stem}-highlight-{highlight.index}.mp4")

    metadata_path = write_metadata(output_dir, input_path, highlights, config, detection_threshold)
    return highlights, metadata_path, detection_threshold


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate chess-stream highlight clips from Twitch VOD audio spikes."
    )
    parser.add_argument("input", type=Path, help="Local Twitch VOD video file, e.g. input.mp4")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("highlights"),
        help="Directory for clips and highlights.json. Default: highlights",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Optional JSON config overriding constants such as AUDIO_THRESHOLD.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze and write JSON metadata without exporting video clips.",
    )
    parser.add_argument(
        "--reencode",
        action="store_true",
        help="Re-encode clips for more accurate cuts instead of fast stream copy.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_path.exists():
        print(f"Input file does not exist: {input_path}", file=sys.stderr)
        return 2

    try:
        config = load_config(args.config.expanduser().resolve() if args.config else None)
        if args.reencode:
            config = Config(**{**asdict(config), "ffmpeg_copy_streams": False})

        highlights, metadata_path, threshold = generate_highlights(
            input_path=input_path,
            output_dir=output_dir,
            config=config,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Detection threshold: {threshold:.6f}")
    print(f"Highlights detected: {len(highlights)}")
    for highlight in highlights:
        print(
            f"{highlight.index:03d}: "
            f"{format_seconds(highlight.start)} -> {format_seconds(highlight.end)} "
            f"peak={format_seconds(highlight.peak_time)} "
            f"confidence={highlight.confidence:.2f}"
        )
    print(f"Metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
