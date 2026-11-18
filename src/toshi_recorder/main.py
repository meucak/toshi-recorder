from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from . import __version__
from .recorder import DEFAULT_RECORDINGS_DIR


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="toshi-recorder",
        description="Toshi Ekran Kayıt — X11 ekran kaydedici ve QR telefon mikrofon stüdyosu.",
    )
    parser.add_argument(
        "--saves",
        action="store_true",
        help="Kayıt klasörünü (~/EkranKayitlari) dosya yöneticisinde açar.",
    )
    parser.add_argument(
        "--saves-path",
        action="store_true",
        help="Kayıt klasörünün yolunu yazdırır.",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="store_true",
        help="Uygulama sürümünü yazdırır.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.version:
        print(f"toshi-recorder {__version__}")
        return 0
    if args.saves_path:
        DEFAULT_RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        print(DEFAULT_RECORDINGS_DIR)
        return 0
    if args.saves:
        DEFAULT_RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["xdg-open", str(DEFAULT_RECORDINGS_DIR)])
        return 0

    from PyQt6.QtWidgets import QApplication
    from .ui import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Toshi Ekran Kayıt")
    app.setOrganizationName("Toshi")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
