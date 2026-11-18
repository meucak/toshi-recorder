from pathlib import Path
import ssl
import urllib.request

from toshi_recorder.phone_mic import (
    PhoneMicBridge,
    decode_audio_packet,
    encode_audio_packet,
    generate_qr_image,
    parse_ipv4_from_ip_addr_output,
    resample_pcm_s16le,
)
from toshi_recorder.recorder import CaptureRegion, build_ffmpeg_command


def test_audio_packet_roundtrip():
    pcm = b"\x01\x00\xff\x7f\x00\x80\x10\x00"
    pkt = encode_audio_packet(48000, pcm)
    decoded = decode_audio_packet(pkt)
    assert decoded == (48000, pcm)
    assert decode_audio_packet(b"BAD0\x00\x00\x00\x00") is None


def test_resample_48k_to_44k():
    # 480 samples at 48kHz (10 ms) -> ~441 samples at 44.1kHz
    pcm = b"\x10\x00" * 480
    out = resample_pcm_s16le(pcm, 48000, 44100)
    assert len(out) == 441 * 2


def test_ip_sorting_prefers_lan_over_tailscale():
    sample = (
        "1: lo    inet 127.0.0.1/8 scope host lo\n"
        "5: tailscale0 inet 100.79.246.19/32 scope global tailscale0\n"
        "3: wlp8s0 inet 192.168.1.27/24 scope global wlp8s0\n"
    )
    ips = parse_ipv4_from_ip_addr_output(sample)
    assert ips == ["192.168.1.27", "100.79.246.19"]


def test_capture_region_even_clamped():
    r = CaptureRegion(x=5, y=7, width=853, height=479).even_clamped()
    assert r.width == 852
    assert r.height == 478
    assert r.x == 5 and r.y == 7


def test_ffmpeg_command_variants(tmp_path: Path):
    fifo = tmp_path / "mic.fifo"
    out = tmp_path / "out.mp4"
    reg = CaptureRegion(0, 0, 1366, 768)

    cmd_mic = build_ffmpeg_command(out, reg, fifo, fps=24, display=":0")
    assert "-f" in cmd_mic and "x11grab" in cmd_mic
    assert "-draw_mouse" in cmd_mic
    assert "pulse" not in cmd_mic
    assert "1366x768" in cmd_mic

    cmd_both = build_ffmpeg_command(
        out,
        reg,
        fifo,
        fps=30,
        display=":0",
        system_monitor_source="alsa_output.pci.monitor",
    )
    assert "pulse" in cmd_both
    assert "alsa_output.pci.monitor" in cmd_both
    assert "amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]" in " ".join(cmd_both)


def test_qr_generation():
    img = generate_qr_image("https://192.168.1.27:8444/?t=abc", scale=4, border=2)
    assert img.ndim == 2
    assert img.shape[0] == img.shape[1]
    assert img.shape[0] > 80


def test_phone_mic_bridge_https_serving(tmp_path: Path):
    bridge = PhoneMicBridge(port=0, cert_dir=tmp_path / "certs", cert_ips=["127.0.0.1"])
    bridge.start()
    try:
        assert bridge.port > 0
        url = bridge.build_url("127.0.0.1")
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=ctx, timeout=3.0) as resp:
            assert resp.status == 200
            html = resp.read().decode("utf-8")
            assert "Toshi Mikrofon" in html
    finally:
        bridge.stop()
