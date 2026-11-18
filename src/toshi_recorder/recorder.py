from __future__ import annotations

from dataclasses import dataclass
import errno
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time
from typing import Optional


DEFAULT_RECORDINGS_DIR = Path.home() / "EkranKayitlari"
TARGET_SAMPLE_RATE = 44100
TICK_INTERVAL = 0.02  # 20 ms
TICK_BYTES = int(TARGET_SAMPLE_RATE * 2 * TICK_INTERVAL)  # 1764 bytes (882 samples)
PREBUFFER_BYTES = TICK_BYTES * 3  # ~60 ms jitter cushion
MAX_BUFFER_BYTES = TICK_BYTES * 20  # ~400 ms max buffer before trimming
TRIM_TARGET_BYTES = TICK_BYTES * 5  # ~100 ms after trim

QUALITY_PRESETS: dict[str, tuple[str, str]] = {
    "balanced": ("ultrafast", "23"),
    "high": ("veryfast", "18"),
    "low_cpu": ("ultrafast", "28"),
}


@dataclass(frozen=True)
class CaptureRegion:
    x: int
    y: int
    width: int
    height: int

    def even_clamped(self) -> "CaptureRegion":
        """Ensure width and height are even and >= 2 for x264 yuv420p."""
        w = max(2, int(self.width) - (int(self.width) % 2))
        h = max(2, int(self.height) - (int(self.height) % 2))
        return CaptureRegion(x=max(0, int(self.x)), y=max(0, int(self.y)), width=w, height=h)


def discover_default_monitor_source() -> Optional[str]:
    """Return PulseAudio/PipeWire monitor source name for desktop output audio."""
    default_sink: Optional[str] = None
    try:
        res = subprocess.run(
            ["pactl", "get-default-sink"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            default_sink = res.stdout.strip()
    except Exception:
        default_sink = None

    try:
        res = subprocess.run(
            ["pactl", "list", "short", "sources"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if res.returncode != 0:
            return f"{default_sink}.monitor" if default_sink else None
        monitors: list[str] = []
        for line in res.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2 and parts[1].endswith(".monitor"):
                monitors.append(parts[1].strip())
        if default_sink and f"{default_sink}.monitor" in monitors:
            return f"{default_sink}.monitor"
        if monitors:
            return monitors[0]
    except Exception:
        pass
    return f"{default_sink}.monitor" if default_sink else None


def build_ffmpeg_command(
    out_path: Path,
    region: CaptureRegion,
    audio_fifo: Path,
    fps: int = 25,
    quality: str = "balanced",
    draw_cursor: bool = True,
    display: Optional[str] = None,
    system_monitor_source: Optional[str] = None,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> list[str]:
    """Build FFmpeg CLI command for X11 screen grab + phone mic FIFO (+ optional system audio)."""
    clean_region = region.even_clamped()
    preset, crf = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["balanced"])
    disp = display or os.environ.get("DISPLAY") or ":0"
    if "." not in disp and ":" in disp:
        disp = f"{disp}.0"

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "x11grab",
        "-thread_queue_size",
        "1024",
        "-framerate",
        str(max(1, int(fps))),
        "-draw_mouse",
        "1" if draw_cursor else "0",
        "-video_size",
        f"{clean_region.width}x{clean_region.height}",
        "-i",
        f"{disp}+{clean_region.x},{clean_region.y}",
        "-f",
        "s16le",
        "-ar",
        str(int(sample_rate)),
        "-ac",
        "1",
        "-thread_queue_size",
        "1024",
        "-i",
        str(audio_fifo),
    ]

    if system_monitor_source:
        cmd.extend(
            [
                "-f",
                "pulse",
                "-thread_queue_size",
                "1024",
                "-i",
                system_monitor_source,
                "-filter_complex",
                (
                    "[1:a]aresample=async=1:first_pts=0[mic];"
                    "[2:a]aresample=async=1:first_pts=0[sys];"
                    "[mic][sys]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]"
                ),
                "-map",
                "0:v",
                "-map",
                "[aout]",
            ]
        )
    else:
        cmd.extend(["-map", "0:v", "-map", "1:a"])

    cmd.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            crf,
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
    )
    return cmd


class ScreenRecorder:
    """Manages an active FFmpeg X11 recording fed by real-time paced phone mic PCM."""

    def __init__(
        self,
        out_path: Path,
        region: CaptureRegion,
        fps: int = 25,
        quality: str = "balanced",
        draw_cursor: bool = True,
        system_monitor_source: Optional[str] = None,
    ) -> None:
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.region = region.even_clamped()
        self.fps = max(1, int(fps))
        self.quality = quality
        self.draw_cursor = draw_cursor
        self.system_monitor_source = system_monitor_source

        self._tmp_dir = tempfile.TemporaryDirectory(prefix="toshi_screenrec_")
        self._audio_fifo = Path(self._tmp_dir.name) / "phone_mic.fifo"
        os.mkfifo(self._audio_fifo)

        self._buf_lock = threading.Lock()
        self._pcm_buf = bytearray()
        self._priming = True
        self._stop_event = threading.Event()
        self._stopped = False

        cmd = build_ffmpeg_command(
            out_path=self.out_path,
            region=self.region,
            audio_fifo=self._audio_fifo,
            fps=self.fps,
            quality=self.quality,
            draw_cursor=self.draw_cursor,
            system_monitor_source=self.system_monitor_source,
        )
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
        self._audio_thread.start()

    def is_running(self) -> bool:
        return not self._stopped and self._proc.poll() is None

    def push_pcm(self, pcm_s16le: bytes) -> None:
        if self._stop_event.is_set() or not pcm_s16le:
            return
        with self._buf_lock:
            self._pcm_buf.extend(pcm_s16le)
            if len(self._pcm_buf) > MAX_BUFFER_BYTES:
                drop = len(self._pcm_buf) - TRIM_TARGET_BYTES
                drop -= drop % 2
                if drop > 0:
                    del self._pcm_buf[:drop]

    def _next_audio_chunk(self, silence: bytes) -> bytes:
        with self._buf_lock:
            buf_len = len(self._pcm_buf)
            if self._priming:
                if buf_len < PREBUFFER_BYTES:
                    return silence
                self._priming = False
            if buf_len >= TICK_BYTES:
                chunk = bytes(self._pcm_buf[:TICK_BYTES])
                del self._pcm_buf[:TICK_BYTES]
                return chunk
            self._priming = True
            return silence

    def _audio_loop(self) -> None:
        fd: Optional[int] = None
        silence = b"\x00" * TICK_BYTES
        next_tick = time.monotonic()
        try:
            while not self._stop_event.is_set():
                if fd is None:
                    if self._proc.poll() is not None:
                        break
                    try:
                        fd = os.open(self._audio_fifo, os.O_WRONLY | os.O_NONBLOCK)
                        next_tick = time.monotonic()
                    except OSError as exc:
                        if exc.errno in (errno.ENXIO, errno.ENOENT):
                            self._stop_event.wait(0.01)
                            continue
                        break

                payload = self._next_audio_chunk(silence)
                offset = 0
                while offset < len(payload) and not self._stop_event.is_set():
                    try:
                        written = os.write(fd, payload[offset:])
                        if written <= 0:
                            break
                        offset += written
                    except BlockingIOError:
                        self._stop_event.wait(0.005)
                    except OSError as exc:
                        if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                            self._stop_event.wait(0.005)
                        elif exc.errno == errno.EPIPE:
                            return
                        else:
                            return

                next_tick += TICK_INTERVAL
                delay = next_tick - time.monotonic()
                if delay > 0:
                    self._stop_event.wait(delay)
                elif delay < -0.25:
                    next_tick = time.monotonic()
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass

    def stop(self) -> tuple[Path, str]:
        """Gracefully finalize recording and return (output_path, stderr_text)."""
        if self._stopped:
            return self.out_path, ""
        self._stopped = True

        if self._proc.poll() is None and self._proc.stdin is not None:
            try:
                self._proc.stdin.write(b"q\n")
                self._proc.stdin.flush()
                self._proc.stdin.close()
            except OSError:
                pass

        if self._proc.poll() is None:
            try:
                self._proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                try:
                    self._proc.send_signal(signal.SIGINT)
                    self._proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    self._proc.terminate()
                    try:
                        self._proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        self._proc.kill()
                        self._proc.wait(timeout=1.0)

        self._stop_event.set()
        self._audio_thread.join(timeout=2.0)

        stderr_text = ""
        if self._proc.stderr is not None:
            try:
                stderr_text = self._proc.stderr.read().decode("utf-8", errors="ignore").strip()
            except Exception:
                stderr_text = ""
        try:
            self._tmp_dir.cleanup()
        except Exception:
            pass

        return self.out_path, stderr_text
