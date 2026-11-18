from __future__ import annotations

import asyncio
from collections import deque
import hmac
import ipaddress
from pathlib import Path
import re
import secrets
import socket
import ssl
import struct
import subprocess
import threading
import time
from typing import Callable, Optional
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np
from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.datastructures import Headers
from websockets.http11 import Request, Response


_INET_RE = re.compile(r"\binet\s+(\d{1,3}(?:\.\d{1,3}){3})/\d+")
_TAILSCALE_NET = ipaddress.IPv4Network("100.64.0.0/10")
AUDIO_MAGIC = b"AUD0"
TARGET_AUDIO_RATE = 44100


def encode_audio_packet(sample_rate: int, pcm_s16le: bytes) -> bytes:
    """Encode mono s16le PCM bytes with magic header and sample rate."""
    return AUDIO_MAGIC + struct.pack("<I", int(sample_rate)) + pcm_s16le


def decode_audio_packet(payload: bytes) -> Optional[tuple[int, bytes]]:
    """Decode AUD0 binary WebSocket audio payload into (sample_rate, pcm_s16le)."""
    if len(payload) < 8 or payload[:4] != AUDIO_MAGIC:
        return None
    (sample_rate,) = struct.unpack("<I", payload[4:8])
    pcm = payload[8:]
    if sample_rate < 8000 or sample_rate > 192000 or len(pcm) < 2 or (len(pcm) % 2) != 0:
        return None
    return int(sample_rate), bytes(pcm)


def resample_pcm_s16le(
    pcm_bytes: bytes, src_rate: int, dst_rate: int = TARGET_AUDIO_RATE
) -> bytes:
    """Linear-resample mono signed 16-bit LE PCM to target sample rate."""
    if not pcm_bytes or src_rate == dst_rate or src_rate <= 0 or dst_rate <= 0:
        return pcm_bytes
    samples = np.frombuffer(pcm_bytes, dtype="<i2")
    if samples.size < 2:
        return pcm_bytes
    dst_len = max(1, int(round(samples.size * (dst_rate / float(src_rate)))))
    x_old = np.linspace(0.0, 1.0, num=samples.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
    interp = np.interp(x_new, x_old, samples.astype(np.float32))
    clipped = np.clip(np.rint(interp), -32768, 32767).astype("<i2")
    return clipped.tobytes()


def _ip_sort_key(ip: str) -> tuple[int, str]:
    addr = ipaddress.IPv4Address(ip)
    if ip.startswith("192.168."):
        return (0, ip)
    if ip.startswith("10."):
        return (1, ip)
    if addr.is_private and addr not in _TAILSCALE_NET:
        return (2, ip)
    if addr in _TAILSCALE_NET:
        return (4, ip)
    return (3, ip)


def parse_ipv4_from_ip_addr_output(output: str) -> list[str]:
    """Extract sorted non-loopback IPv4 addresses from `ip -4 -o addr show`."""
    found: list[str] = []
    seen: set[str] = set()
    for match in _INET_RE.finditer(output or ""):
        ip = match.group(1)
        try:
            addr = ipaddress.IPv4Address(ip)
        except ipaddress.AddressValueError:
            continue
        if addr.is_loopback or addr.is_link_local or addr.is_unspecified:
            continue
        if ip not in seen:
            seen.add(ip)
            found.append(ip)
    found.sort(key=_ip_sort_key)
    return found


def discover_lan_ips() -> list[str]:
    """Discover local IPv4 addresses preferring Wi-Fi/LAN subnets over Tailscale."""
    ips: list[str] = []
    try:
        res = subprocess.run(
            ["ip", "-4", "-o", "addr", "show"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if res.returncode == 0:
            ips = parse_ipv4_from_ip_addr_output(res.stdout)
    except Exception:
        ips = []

    if not ips:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("192.168.1.1", 80))
                local_ip = s.getsockname()[0]
                if local_ip and not local_ip.startswith("127."):
                    ips = [local_ip]
        except OSError:
            pass

    return ips or ["127.0.0.1"]


def ensure_self_signed_cert(cert_dir: Path, ips: list[str]) -> tuple[Path, Path]:
    """Create or reuse a self-signed TLS certificate containing LAN IP SANs."""
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert_path = cert_dir / "cert.pem"
    key_path = cert_dir / "key.pem"
    marker_path = cert_dir / ".san_ips"

    clean_ips = sorted({ip.strip() for ip in ips if ip.strip()})
    marker_val = ",".join(clean_ips)

    if (
        cert_path.exists()
        and key_path.exists()
        and marker_path.exists()
        and marker_path.read_text(encoding="utf-8", errors="ignore").strip() == marker_val
    ):
        return cert_path, key_path

    san_entries = ["DNS:localhost", "IP:127.0.0.1"]
    for ip in clean_ips:
        if ip != "127.0.0.1":
            san_entries.append(f"IP:{ip}")
    san_ext = f"subjectAltName={','.join(san_entries)}"

    cmd = [
        "openssl",
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-sha256",
        "-days",
        "365",
        "-keyout",
        str(key_path),
        "-out",
        str(cert_path),
        "-subj",
        "/CN=Toshi Ekran Kayit",
        "-addext",
        san_ext,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=15.0, check=False)
    if res.returncode != 0 or not cert_path.exists() or not key_path.exists():
        raise RuntimeError(f"TLS sertifikası oluşturulamadı: {res.stderr.strip()}")

    marker_path.write_text(marker_val, encoding="utf-8")
    return cert_path, key_path


def generate_qr_image(url: str, scale: int = 8, border: int = 4) -> np.ndarray:
    """Render an exact crisp QR matrix as a grayscale uint8 NumPy array."""
    encoder = cv2.QRCodeEncoder.create()
    matrix = encoder.encode(url)
    if border > 0:
        matrix = cv2.copyMakeBorder(
            matrix,
            border,
            border,
            border,
            border,
            cv2.BORDER_CONSTANT,
            value=255,
        )
    if scale > 1:
        h, w = matrix.shape[:2]
        matrix = cv2.resize(
            matrix, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST
        )
    return matrix


def build_mobile_page_html() -> str:
    """Return mobile microphone web bridge page served over HTTPS."""
    return """<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no" />
  <title>Toshi Ekran Kayıt — Telefon Mikrofonu</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #111317;
      --card: #1b1e24;
      --border: #2c313c;
      --text: #f3f5f8;
      --muted: #9aa3b2;
      --accent: #0a84ff;
      --ok: #30d158;
      --warn: #ff9f0a;
      --err: #ff453a;
    }
    * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
    html, body {
      margin: 0; padding: 0; width: 100%; min-height: 100%;
      background: var(--bg); color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    body {
      display: flex; align-items: center; justify-content: center;
      padding: 18px;
    }
    .panel {
      width: 100%; max-width: 440px;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 20px;
      display: flex; flex-direction: column; gap: 16px;
    }
    .head {
      display: flex; align-items: center; justify-content: space-between; gap: 12px;
    }
    .title {
      font-size: 18px; font-weight: 700; margin: 0;
    }
    .sub {
      font-size: 13px; color: var(--muted); margin-top: 3px;
    }
    .badge {
      display: inline-flex; align-items: center; gap: 8px;
      padding: 7px 12px; border-radius: 999px;
      background: #13161b; border: 1px solid var(--border);
      font-size: 13px; font-weight: 600;
    }
    .dot {
      width: 10px; height: 10px; border-radius: 50%;
      background: var(--warn); flex-shrink: 0;
    }
    .dot.ok { background: var(--ok); }
    .dot.err { background: var(--err); }
    .meter-box {
      background: #13161b; border: 1px solid var(--border);
      border-radius: 14px; padding: 14px;
      display: flex; flex-direction: column; gap: 10px;
    }
    .meter-row {
      display: flex; justify-content: space-between; align-items: center;
      font-size: 13px; color: var(--muted); font-variant-numeric: tabular-nums;
    }
    .bar-bg {
      width: 100%; height: 18px; border-radius: 9px;
      background: #232731; overflow: hidden;
    }
    .bar-fill {
      width: 0%; height: 100%;
      background: linear-gradient(90deg, #30d158 0%, #ff9f0a 75%, #ff453a 100%);
      transition: width 60ms linear;
    }
    .controls {
      display: flex; flex-direction: column; gap: 10px;
    }
    button {
      width: 100%; min-height: 52px; border-radius: 14px;
      border: 1px solid var(--border); background: #242933;
      color: var(--text); font-size: 16px; font-weight: 600;
      cursor: pointer; padding: 12px 16px;
    }
    button.primary {
      background: var(--accent); border-color: var(--accent); color: #fff;
    }
    button.danger {
      background: rgba(255, 69, 58, 0.18); border-color: var(--err); color: #ff8a80;
    }
    button:active { opacity: 0.85; }
    .slider-card {
      background: #13161b; border: 1px solid var(--border);
      border-radius: 14px; padding: 12px 14px;
      display: flex; flex-direction: column; gap: 8px;
    }
    .slider-card label {
      display: flex; justify-content: space-between;
      font-size: 14px; font-weight: 500;
    }
    input[type="range"] { width: 100%; accent-color: var(--accent); height: 28px; }
    .err-box {
      display: none; padding: 12px 14px; border-radius: 12px;
      background: rgba(255, 69, 58, 0.16); border: 1px solid var(--err);
      color: #ffb4ae; font-size: 13px; line-height: 1.4;
    }
    .foot {
      font-size: 12px; color: var(--muted); line-height: 1.45;
    }
  </style>
</head>
<body>
  <main class="panel">
    <div class="head">
      <div>
        <h1 class="title">Toshi Mikrofon</h1>
        <div class="sub">Yerel Ağ Ses Köprüsü</div>
      </div>
      <div class="badge">
        <span class="dot" id="dot"></span>
        <span id="statusText">Hazırlanıyor…</span>
      </div>
    </div>

    <div class="meter-box">
      <div class="meter-row">
        <span>Mikrofon Seviyesi</span>
        <span id="metaText">Bekleniyor</span>
      </div>
      <div class="bar-bg">
        <div class="bar-fill" id="levelFill"></div>
      </div>
    </div>

    <div class="slider-card">
      <label for="gainRange">
        <span>Ses Kazancı (Gain)</span>
        <span id="gainLabel">1.2x</span>
      </label>
      <input id="gainRange" type="range" min="0.5" max="3.0" step="0.1" value="1.2" />
    </div>

    <div class="controls">
      <button class="primary" id="startBtn">Mikrofonu Başlat</button>
      <button id="muteBtn" disabled>Mikrofon Açık</button>
      <button id="nsBtn">Gürültü Engelleme: Açık</button>
    </div>

    <div class="err-box" id="errBox"></div>

    <div class="foot">
      Ekran kaydı süresince bu sekmenin açık kalması yeterlidir. Telefon uyku kilidi otomatik engellenir.
    </div>
  </main>

  <script>
    (() => {
      const qs = new URLSearchParams(window.location.search);
      const token = qs.get("t") || "";

      const dot = document.getElementById("dot");
      const statusText = document.getElementById("statusText");
      const metaText = document.getElementById("metaText");
      const levelFill = document.getElementById("levelFill");
      const startBtn = document.getElementById("startBtn");
      const muteBtn = document.getElementById("muteBtn");
      const nsBtn = document.getElementById("nsBtn");
      const gainRange = document.getElementById("gainRange");
      const gainLabel = document.getElementById("gainLabel");
      const errBox = document.getElementById("errBox");

      let ws = null;
      let stream = null;
      let audioCtx = null;
      let sourceNode = null;
      let gainNode = null;
      let procNode = null;
      let silentGain = null;
      let wakeLock = null;
      let reconnectTimer = null;

      let micMuted = false;
      let noiseSuppress = true;
      let gainValue = 1.2;
      let packetsSent = 0;

      function setStatus(text, kind) {
        statusText.textContent = text;
        dot.className = "dot" + (kind ? " " + kind : "");
      }

      function showError(msg) {
        errBox.textContent = msg;
        errBox.style.display = "block";
      }

      function hideError() {
        errBox.style.display = "none";
      }

      function updateStatusUI() {
        const wsOk = ws && ws.readyState === WebSocket.OPEN;
        const audioOk = audioCtx && audioCtx.state === "running" && stream;
        if (!wsOk) {
          setStatus("Toshi'ye bağlanılıyor…", "");
          return;
        }
        if (!audioOk) {
          setStatus("Dokunup Mikrofonu Başlat", "");
          return;
        }
        if (micMuted) {
          setStatus("Sessizde (Mute)", "");
          return;
        }
        setStatus("Canlı — Ses Aktarılıyor", "ok");
      }

      async function acquireWakeLock() {
        if (!("wakeLock" in navigator)) return;
        try {
          if (!wakeLock) {
            wakeLock = await navigator.wakeLock.request("screen");
            wakeLock.addEventListener("release", () => { wakeLock = null; });
          }
        } catch (_) {}
      }

      function connectWs() {
        if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
          return;
        }
        const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = proto + "//" + window.location.host + "/ws?t=" + encodeURIComponent(token);
        ws = new WebSocket(wsUrl);
        ws.binaryType = "arraybuffer";
        ws.onopen = () => updateStatusUI();
        ws.onclose = () => {
          updateStatusUI();
          clearTimeout(reconnectTimer);
          reconnectTimer = setTimeout(connectWs, 1000);
        };
        ws.onerror = () => {
          setStatus("WebSocket hatası", "err");
        };
      }

      function teardownAudio() {
        if (procNode) { try { procNode.disconnect(); } catch (_) {} procNode = null; }
        if (gainNode) { try { gainNode.disconnect(); } catch (_) {} gainNode = null; }
        if (sourceNode) { try { sourceNode.disconnect(); } catch (_) {} sourceNode = null; }
        if (silentGain) { try { silentGain.disconnect(); } catch (_) {} silentGain = null; }
        if (audioCtx) { try { audioCtx.close(); } catch (_) {} audioCtx = null; }
        if (stream) {
          stream.getTracks().forEach(t => t.stop());
          stream = null;
        }
      }

      async function startMic() {
        hideError();
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          setStatus("HTTPS gerekli", "err");
          showError("Tarayıcı mikrofon izni veremedi. HTTPS güvenlik uyarısını onayladığınızdan emin olun.");
          return;
        }
        teardownAudio();
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            audio: {
              echoCancellation: noiseSuppress,
              noiseSuppression: noiseSuppress,
              autoGainControl: noiseSuppress,
              channelCount: 1
            },
            video: false
          });
          const AudioCtx = window.AudioContext || window.webkitAudioContext;
          try {
            audioCtx = new AudioCtx({ sampleRate: 44100 });
          } catch (_) {
            audioCtx = new AudioCtx();
          }
          if (audioCtx.state === "suspended") {
            await audioCtx.resume();
          }
          sourceNode = audioCtx.createMediaStreamSource(stream);
          gainNode = audioCtx.createGain();
          gainNode.gain.value = gainValue;

          procNode = audioCtx.createScriptProcessor(2048, 1, 1);
          silentGain = audioCtx.createGain();
          silentGain.gain.value = 0;

          procNode.onaudioprocess = (ev) => {
            const input = ev.inputBuffer.getChannelData(0);
            let sumSq = 0;
            for (let i = 0; i < input.length; i++) {
              sumSq += input[i] * input[i];
            }
            const rms = Math.sqrt(sumSq / Math.max(1, input.length));
            const pct = micMuted ? 0 : Math.min(100, Math.round(rms * 260));
            levelFill.style.width = pct + "%";

            if (micMuted || !ws || ws.readyState !== WebSocket.OPEN) {
              return;
            }
            const out = new ArrayBuffer(8 + input.length * 2);
            const view = new DataView(out);
            view.setUint8(0, 65); // 'A'
            view.setUint8(1, 85); // 'U'
            view.setUint8(2, 68); // 'D'
            view.setUint8(3, 48); // '0'
            view.setUint32(4, audioCtx.sampleRate || 44100, true);
            for (let i = 0; i < input.length; i++) {
              const s = Math.max(-1, Math.min(1, input[i]));
              view.setInt16(8 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
            }
            ws.send(out);
            packetsSent += 1;
            if (packetsSent % 10 === 1) {
              metaText.textContent = (audioCtx.sampleRate || 44100) + " Hz • Canlı";
            }
          };

          sourceNode.connect(gainNode);
          gainNode.connect(procNode);
          procNode.connect(silentGain);
          silentGain.connect(audioCtx.destination);

          startBtn.textContent = "Mikrofonu Yeniden Başlat";
          muteBtn.disabled = false;
          acquireWakeLock();
          connectWs();
          updateStatusUI();
        } catch (err) {
          setStatus("Mikrofon başlatılamadı", "err");
          showError("Mikrofon izni alınamadı: " + (err && err.message ? err.message : err));
        }
      }

      startBtn.addEventListener("click", () => startMic());

      muteBtn.addEventListener("click", () => {
        micMuted = !micMuted;
        muteBtn.textContent = micMuted ? "Sessizde (Açmak İçin Dokun)" : "Mikrofon Açık";
        muteBtn.className = micMuted ? "danger" : "";
        if (audioCtx && audioCtx.state === "suspended" && !micMuted) {
          audioCtx.resume().catch(() => {});
        }
        updateStatusUI();
      });

      nsBtn.addEventListener("click", () => {
        noiseSuppress = !noiseSuppress;
        nsBtn.textContent = "Gürültü Engelleme: " + (noiseSuppress ? "Açık" : "Kapalı");
        if (stream) {
          startMic();
        }
      });

      gainRange.addEventListener("input", () => {
        gainValue = parseFloat(gainRange.value) || 1.0;
        gainLabel.textContent = gainValue.toFixed(1) + "x";
        if (gainNode) {
          gainNode.gain.value = gainValue;
        }
      });

      document.addEventListener("visibilitychange", () => {
        if (!document.hidden) {
          acquireWakeLock();
          connectWs();
          if (audioCtx && audioCtx.state === "suspended") {
            audioCtx.resume().catch(() => {});
          }
        }
      });

      connectWs();
      startMic();
    })();
  </script>
</body>
</html>
"""


class PhoneMicBridge:
    """Single-port HTTPS + WSS server receiving phone microphone PCM audio."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8444,
        cert_dir: Optional[Path] = None,
        cert_ips: Optional[list[str]] = None,
    ) -> None:
        self.host = host
        self.requested_port = port
        self.cert_dir = cert_dir or (Path.home() / ".config" / "toshi-recorder" / "certs")
        self._cert_ips = cert_ips
        self.token = secrets.token_urlsafe(16)

        self._lock = threading.Lock()
        self._audio_callback: Optional[Callable[[bytes], None]] = None
        self._last_audio_time: float = 0.0
        self._audio_level: float = 0.0
        self._active_clients: int = 0
        self._client_addr: Optional[str] = None
        self._total_packets: int = 0

        self._bound_port: int = 0
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event_async: Optional[asyncio.Event] = None
        self._ready_event = threading.Event()
        self._start_error: Optional[BaseException] = None
        self._running = False
        self._html_bytes = build_mobile_page_html().encode("utf-8")

    @property
    def port(self) -> int:
        return self._bound_port

    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    def build_url(self, ip: str) -> str:
        return f"https://{ip}:{self._bound_port}/?t={self.token}"

    def regenerate_token(self) -> str:
        with self._lock:
            self.token = secrets.token_urlsafe(16)
            return self.token

    def set_audio_callback(self, cb: Optional[Callable[[bytes], None]]) -> None:
        with self._lock:
            self._audio_callback = cb

    def is_audio_active(self) -> bool:
        now = time.monotonic()
        with self._lock:
            return self._last_audio_time > 0.0 and (now - self._last_audio_time) <= 1.2

    def audio_level(self) -> float:
        if not self.is_audio_active():
            return 0.0
        with self._lock:
            return self._audio_level

    def is_client_connected(self) -> bool:
        with self._lock:
            return self._active_clients > 0

    def client_info(self) -> Optional[str]:
        with self._lock:
            if self._active_clients <= 0:
                return None
            return self._client_addr

    def start(self) -> None:
        if self.is_running():
            return
        ips = self._cert_ips if self._cert_ips is not None else discover_lan_ips()
        cert_path, key_path = ensure_self_signed_cert(self.cert_dir, ips)

        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))

        self._ready_event.clear()
        self._start_error = None
        self._thread = threading.Thread(
            target=self._run_loop, args=(ssl_ctx,), daemon=True
        )
        self._thread.start()
        self._ready_event.wait(timeout=5.0)
        if self._start_error is not None:
            raise RuntimeError(f"Telefon mikrofon sunucusu başlatılamadı: {self._start_error}")
        if not self._running:
            raise RuntimeError("Telefon mikrofon sunucusu zaman aşımına uğradı")

    def stop(self) -> None:
        loop = self._loop
        stop_ev = self._stop_event_async
        if loop is not None and stop_ev is not None and not loop.is_closed():
            loop.call_soon_threadsafe(stop_ev.set)
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._running = False
        with self._lock:
            self._active_clients = 0
            self._client_addr = None

    def _token_valid(self, query: str) -> bool:
        params = parse_qs(query, keep_blank_values=False)
        supplied = params.get("t", [""])[0]
        with self._lock:
            expected = self.token
        return bool(supplied) and hmac.compare_digest(supplied, expected)

    def _process_request(
        self, connection: ServerConnection, request: Request
    ) -> Optional[Response]:
        parsed = urlsplit(request.path)
        if parsed.path in ("/", "/index.html"):
            if not self._token_valid(parsed.query):
                return Response(
                    403,
                    "Forbidden",
                    Headers([("Content-Type", "text/plain; charset=utf-8")]),
                    "Yetkisiz erisim: gecersiz oturum anahtari.".encode("utf-8"),
                )
            return Response(
                200,
                "OK",
                Headers(
                    [
                        ("Content-Type", "text/html; charset=utf-8"),
                        ("Cache-Control", "no-store"),
                    ]
                ),
                self._html_bytes,
            )

        if parsed.path == "/ws":
            if not self._token_valid(parsed.query):
                return Response(
                    403,
                    "Forbidden",
                    Headers([("Content-Type", "text/plain; charset=utf-8")]),
                    b"Unauthorized WebSocket token",
                )
            return None

        return Response(
            404,
            "Not Found",
            Headers([("Content-Type", "text/plain; charset=utf-8")]),
            b"Not Found",
        )

    async def _ws_handler(self, ws: ServerConnection) -> None:
        remote = ws.remote_address
        addr_str = f"{remote[0]}" if isinstance(remote, tuple) and remote else "telefon"
        with self._lock:
            self._active_clients += 1
            self._client_addr = addr_str
        try:
            async for message in ws:
                if not isinstance(message, (bytes, bytearray)) or not message:
                    continue
                raw_bytes = bytes(message)
                parsed_audio = decode_audio_packet(raw_bytes)
                if parsed_audio is None:
                    continue
                src_rate, pcm = parsed_audio
                pcm_44k = resample_pcm_s16le(pcm, src_rate, TARGET_AUDIO_RATE)
                samples = np.frombuffer(pcm_44k, dtype="<i2")
                rms = (
                    float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)) / 32768.0)
                    if samples.size > 0
                    else 0.0
                )
                now = time.monotonic()
                cb: Optional[Callable[[bytes], None]] = None
                with self._lock:
                    self._last_audio_time = now
                    self._audio_level = min(1.0, rms * 2.5)
                    self._total_packets += 1
                    cb = self._audio_callback
                if cb is not None:
                    try:
                        cb(pcm_44k)
                    except Exception:
                        pass
        finally:
            with self._lock:
                self._active_clients = max(0, self._active_clients - 1)
                if self._active_clients == 0:
                    self._client_addr = None

    async def _serve_async(self, ssl_ctx: ssl.SSLContext) -> None:
        self._stop_event_async = asyncio.Event()
        candidate_ports = (
            [0]
            if self.requested_port == 0
            else list(range(self.requested_port, self.requested_port + 10)) + [0]
        )

        server: Optional[Server] = None
        last_err: Optional[OSError] = None
        for p in candidate_ports:
            try:
                server = await serve(
                    self._ws_handler,
                    self.host,
                    p,
                    ssl=ssl_ctx,
                    process_request=self._process_request,
                    max_size=2 * 1024 * 1024,
                    ping_interval=15,
                    ping_timeout=15,
                )
                break
            except OSError as exc:
                last_err = exc

        if server is None:
            raise last_err or OSError("Uygun port bulunamadı")

        async with server:
            sock = server.sockets[0]
            self._bound_port = int(sock.getsockname()[1])
            self._running = True
            self._ready_event.set()
            await self._stop_event_async.wait()

    def _run_loop(self, ssl_ctx: ssl.SSLContext) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._serve_async(ssl_ctx))
        except BaseException as exc:
            self._start_error = exc
            self._running = False
            self._ready_event.set()
        finally:
            self._running = False
            loop.close()
