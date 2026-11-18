from __future__ import annotations

from datetime import datetime
from pathlib import Path
import subprocess
import time
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QRect, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QColor,
    QFont,
    QGuiApplication,
    QIcon,
    QImage,
    QKeySequence,
    QPainter,
    QPalette,
    QPen,
    QPixmap,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from .phone_mic import PhoneMicBridge, discover_lan_ips, generate_qr_image
from .recorder import (
    DEFAULT_RECORDINGS_DIR,
    CaptureRegion,
    ScreenRecorder,
    discover_default_monitor_source,
)


def _numpy_gray_to_pixmap(img: np.ndarray) -> QPixmap:
    h, w = img.shape[:2]
    contiguous = np.ascontiguousarray(img, dtype=np.uint8)
    qimg = QImage(contiguous.data, w, h, w, QImage.Format.Format_Grayscale8)
    return QPixmap.fromImage(qimg.copy())


def _theme_icon(name: str, fallback_text: str = "") -> QIcon:
    icon = QIcon.fromTheme(name)
    if not icon.isNull():
        return icon
    pixmap = QPixmap(24, 24)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor("#d70015") if "record" in name else QColor("#0a84ff"))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(3, 3, 18, 18)
    painter.end()
    return QIcon(pixmap)


def _make_tray_icon(recording: bool) -> QIcon:
    pixmap = QPixmap(24, 24)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    outer = QColor("#d70015") if recording else QColor("#0a84ff")
    painter.setBrush(outer)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRoundedRect(2, 4, 20, 16, 4, 4)
    painter.setBrush(QColor("#ffffff"))
    if recording:
        painter.drawRect(8, 8, 8, 8)
    else:
        painter.drawEllipse(8, 8, 8, 8)
    painter.end()
    return QIcon(pixmap)


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hrs, rem = divmod(total, 3600)
    mins, secs = divmod(rem, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def _format_size(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        return "0.0 MB"
    if size < 1024 * 1024:
        return f"{size / 1024:.0f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


class RegionSelectorOverlay(QWidget):
    """Compositor-safe fullscreen region picker using a frozen screen snapshot."""

    region_selected = pyqtSignal(CaptureRegion)
    cancelled = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(
            parent,
            Qt.WindowType.Window
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        screen = QGuiApplication.primaryScreen()
        self._screen_rect = screen.geometry() if screen else QRect(0, 0, 1366, 768)
        self._snapshot = screen.grabWindow(0) if screen else QPixmap()
        self._origin: Optional[QPoint] = None
        self._current: Optional[QPoint] = None
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)
        self.setGeometry(self._screen_rect)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            self.cancelled.emit()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            self.close()
            self.cancelled.emit()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._origin = event.position().toPoint()
            self._current = self._origin
            self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._origin is not None:
            self._current = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._origin is not None:
            end_pt = event.position().toPoint()
            rect = QRect(self._origin, end_pt).normalized()
            self._origin = None
            self._current = None
            self.close()
            if rect.width() >= 32 and rect.height() >= 32:
                region = CaptureRegion(
                    x=self._screen_rect.x() + rect.x(),
                    y=self._screen_rect.y() + rect.y(),
                    width=rect.width(),
                    height=rect.height(),
                ).even_clamped()
                self.region_selected.emit(region)
            else:
                self.cancelled.emit()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        if not self._snapshot.isNull():
            painter.drawPixmap(0, 0, self._snapshot)

        dim = QColor(0, 0, 0, 125)
        w = self.width()
        h = self.height()

        if self._origin is not None and self._current is not None:
            sel = QRect(self._origin, self._current).normalized().intersected(self.rect())
            # Draw 4 dimming bands around selection so it works with zero compositor
            painter.fillRect(0, 0, w, sel.top(), dim)
            painter.fillRect(0, sel.bottom() + 1, w, h - sel.bottom() - 1, dim)
            painter.fillRect(0, sel.top(), sel.left(), sel.height(), dim)
            painter.fillRect(sel.right() + 1, sel.top(), w - sel.right() - 1, sel.height(), dim)

            painter.setPen(QPen(QColor("#0a84ff"), 2))
            painter.drawRect(sel)

            even_w = max(2, sel.width() - (sel.width() % 2))
            even_h = max(2, sel.height() - (sel.height() % 2))
            badge_text = f"{even_w} × {even_h}"
            badge_rect = QRect(sel.left(), max(8, sel.top() - 30), 116, 24)
            painter.fillRect(badge_rect, QColor(17, 19, 23, 220))
            painter.setPen(QColor("#ffffff"))
            painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, badge_text)
        else:
            painter.fillRect(self.rect(), dim)
            tip = QRect(0, 24, w, 36)
            painter.setPen(QColor("#ffffff"))
            font = painter.font()
            font.setPointSize(12)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(
                tip,
                Qt.AlignmentFlag.AlignCenter,
                "Kaydedilecek alanı sürükleyip seçin • İptal için Esc",
            )


class MainWindow(QMainWindow):
    """Toshi Ekran Kayıt — Masaüstü ekran kaydedici ve QR telefon mikrofon stüdyosu."""

    def __init__(self, recordings_dir: Path = DEFAULT_RECORDINGS_DIR) -> None:
        super().__init__()
        self.recordings_dir = Path(recordings_dir)
        self.recordings_dir.mkdir(parents=True, exist_ok=True)

        self.setWindowTitle("Toshi Ekran Kayıt")
        self.setWindowIcon(_theme_icon("media-record"))

        screen = QGuiApplication.primaryScreen()
        geom = screen.geometry() if screen else QRect(0, 0, 1366, 768)
        self._full_region = CaptureRegion(
            x=geom.x(), y=geom.y(), width=geom.width(), height=geom.height()
        ).even_clamped()
        self._selected_region = self._full_region
        self._use_custom_region = False

        self._recorder: Optional[ScreenRecorder] = None
        self._recording_started_at: float = 0.0
        self._last_saved_path: Optional[Path] = None
        self._smoothed_level: float = 0.0
        self._region_overlay: Optional[RegionSelectorOverlay] = None
        self._force_quit = False

        self._bridge = PhoneMicBridge(port=8444)
        self._bridge.set_audio_callback(self._on_phone_pcm)
        self._bridge.start()

        self._build_ui()
        self._setup_tray()
        self._refresh_qr()

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(80)
        self._poll_timer.timeout.connect(self._on_tick)
        self._poll_timer.start()

        self.resize(840, 520)

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # Header
        header_row = QHBoxLayout()
        header_row.setSpacing(12)

        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        title_lbl = QLabel("Toshi Ekran Kayıt")
        title_font = QFont(title_lbl.font())
        title_font.setPointSize(title_font.pointSize() + 5)
        title_font.setBold(True)
        title_lbl.setFont(title_font)

        sub_lbl = QLabel("X11 ekran kaydı ve uygulamasız QR telefon mikrofon stüdyosu")
        sub_lbl.setWordWrap(True)
        sub_pal = sub_lbl.palette()
        sub_pal.setColor(QPalette.ColorRole.WindowText, self.palette().color(QPalette.ColorRole.PlaceholderText))
        sub_lbl.setPalette(sub_pal)

        title_col.addWidget(title_lbl)
        title_col.addWidget(sub_lbl)
        header_row.addLayout(title_col, 1)

        self.state_badge = QLabel("Hazır")
        self.state_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.state_badge.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._set_badge_style(recording=False)
        header_row.addWidget(self.state_badge)
        root.addLayout(header_row)

        # Body 2-Column Split
        body = QHBoxLayout()
        body.setSpacing(16)
        root.addLayout(body, 1)

        # Left Pane: Studio Controls
        left_col = QVBoxLayout()
        left_col.setSpacing(12)
        body.addLayout(left_col, 3)

        # Card 1: Capture & Video Options
        video_card = self._make_card()
        video_layout = QVBoxLayout(video_card)
        video_layout.setContentsMargins(14, 14, 14, 14)
        video_layout.setSpacing(10)
        video_layout.addWidget(self._section_title("Görüntü ve Kayıt Ayarları"))

        form = QFormLayout()
        form.setSpacing(8)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        region_row = QHBoxLayout()
        region_row.setSpacing(8)
        self.region_combo = QComboBox()
        self.region_combo.addItem(
            f"Tam Ekran ({self._full_region.width}×{self._full_region.height})", "full"
        )
        self.region_combo.addItem("Özel Bölge", "custom")
        self.region_combo.currentIndexChanged.connect(self._on_region_mode_changed)

        self.pick_region_btn = QPushButton("&Bölge Seç…")
        self.pick_region_btn.setIcon(_theme_icon("view-fullscreen"))
        self.pick_region_btn.clicked.connect(self._open_region_picker)

        region_row.addWidget(self.region_combo, 1)
        region_row.addWidget(self.pick_region_btn)

        region_lbl = QLabel("Yakala&ma alanı")
        region_lbl.setBuddy(self.region_combo)
        form.addRow(region_lbl, region_row)

        self.region_info_lbl = QLabel(self._format_region_text())
        self.region_info_lbl.setWordWrap(True)
        form.addRow("", self.region_info_lbl)

        self.fps_combo = QComboBox()
        self.fps_combo.addItem("15 FPS (En Hafif)", 15)
        self.fps_combo.addItem("24 FPS (Dengeli)", 24)
        self.fps_combo.addItem("30 FPS (Akıcı)", 30)
        self.fps_combo.setCurrentIndex(1)
        fps_lbl = QLabel("Ka&re hızı")
        fps_lbl.setBuddy(self.fps_combo)
        form.addRow(fps_lbl, self.fps_combo)

        self.quality_combo = QComboBox()
        self.quality_combo.addItem("Dengeli (Önerilen)", "balanced")
        self.quality_combo.addItem("Yüksek Kalite", "high")
        self.quality_combo.addItem("Düşük CPU", "low_cpu")
        qual_lbl = QLabel("Ka&lite")
        qual_lbl.setBuddy(self.quality_combo)
        form.addRow(qual_lbl, self.quality_combo)

        video_layout.addLayout(form)

        self.cursor_chk = QCheckBox("Fare imlecini göster")
        self.cursor_chk.setChecked(True)
        self.system_audio_chk = QCheckBox("Bilgisayar sesini (hoparlör) dahil et")
        self.system_audio_chk.setChecked(False)
        self.minimize_chk = QCheckBox("Kayıt başlayınca pencereyi simge durumuna küçült")
        self.minimize_chk.setChecked(True)

        video_layout.addWidget(self.cursor_chk)
        video_layout.addWidget(self.system_audio_chk)
        video_layout.addWidget(self.minimize_chk)
        left_col.addWidget(video_card)

        # Card 2: Phone Mic Status & Level Meter
        mic_card = self._make_card()
        mic_layout = QVBoxLayout(mic_card)
        mic_layout.setContentsMargins(14, 14, 14, 14)
        mic_layout.setSpacing(8)
        mic_layout.addWidget(self._section_title("Telefon Mikrofonu"))

        mic_status_row = QHBoxLayout()
        mic_status_row.setSpacing(8)
        self.mic_dot = QLabel("●")
        self.mic_status_lbl = QLabel("Telefon bekleniyor…")
        self.mic_status_lbl.setWordWrap(True)
        mic_status_row.addWidget(self.mic_dot)
        mic_status_row.addWidget(self.mic_status_lbl, 1)
        mic_layout.addLayout(mic_status_row)

        self.vu_bar = QProgressBar()
        self.vu_bar.setRange(0, 100)
        self.vu_bar.setValue(0)
        self.vu_bar.setTextVisible(False)
        self.vu_bar.setMinimumHeight(14)
        self.vu_bar.setAccessibleName("Mikrofon Ses Seviyesi")
        mic_layout.addWidget(self.vu_bar)

        mic_hint = QLabel(
            "Telefon bağlı değilken kayıt başlatırsanız ses kanalı sessiz başlar; telefon bağlandığı anda ses otomatik eklenir."
        )
        mic_hint.setWordWrap(True)
        mic_hint.setPalette(sub_pal)
        mic_layout.addWidget(mic_hint)
        left_col.addWidget(mic_card)

        # Card 3: Recording Transport & Output
        rec_card = self._make_card()
        rec_layout = QVBoxLayout(rec_card)
        rec_layout.setContentsMargins(14, 14, 14, 14)
        rec_layout.setSpacing(10)

        stats_row = QHBoxLayout()
        self.timer_lbl = QLabel("00:00:00")
        t_font = QFont(self.timer_lbl.font())
        t_font.setPointSize(t_font.pointSize() + 8)
        t_font.setBold(True)
        t_font.setStyleHint(QFont.StyleHint.Monospace)
        self.timer_lbl.setFont(t_font)

        self.file_meta_lbl = QLabel(f"Kayıt dizini: {self.recordings_dir}")
        self.file_meta_lbl.setWordWrap(True)
        self.file_meta_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        stats_row.addWidget(self.timer_lbl)
        stats_row.addWidget(self.file_meta_lbl, 1)
        rec_layout.addLayout(stats_row)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.record_btn = QPushButton("&Kaydı Başlat (F9)")
        self.record_btn.setIcon(_theme_icon("media-record"))
        self.record_btn.setMinimumHeight(42)
        self.record_btn.setDefault(True)
        self._apply_record_btn_style(recording=False)
        self.record_btn.clicked.connect(self._toggle_recording)

        self.open_dir_btn = QPushButton("Klasörü &Aç")
        self.open_dir_btn.setIcon(_theme_icon("folder-open"))
        self.open_dir_btn.setMinimumHeight(42)
        self.open_dir_btn.clicked.connect(self._open_recordings_dir)

        self.play_last_btn = QPushButton("Son Kaydı &Oynat")
        self.play_last_btn.setIcon(_theme_icon("media-playback-start"))
        self.play_last_btn.setMinimumHeight(42)
        self.play_last_btn.setEnabled(False)
        self.play_last_btn.clicked.connect(self._play_last_recording)

        btn_row.addWidget(self.record_btn, 2)
        btn_row.addWidget(self.open_dir_btn, 1)
        btn_row.addWidget(self.play_last_btn, 1)
        rec_layout.addLayout(btn_row)
        left_col.addWidget(rec_card)
        left_col.addStretch(1)

        # Right Pane: QR Bridge
        qr_card = self._make_card()
        qr_layout = QVBoxLayout(qr_card)
        qr_layout.setContentsMargins(16, 16, 16, 16)
        qr_layout.setSpacing(10)
        body.addWidget(qr_card, 2)

        qr_layout.addWidget(self._section_title("QR Kod ile Telefon Bağla"))

        self.qr_label = QLabel()
        self.qr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.qr_label.setMinimumSize(210, 210)
        self.qr_label.setStyleSheet(
            "background: #ffffff; border: 1px solid #d0d4dc; border-radius: 10px; padding: 8px;"
        )
        qr_layout.addWidget(self.qr_label, 0, Qt.AlignmentFlag.AlignCenter)

        ip_form = QFormLayout()
        ip_form.setSpacing(6)
        self.ip_combo = QComboBox()
        for ip in discover_lan_ips():
            self.ip_combo.addItem(ip, ip)
        self.ip_combo.currentIndexChanged.connect(self._refresh_qr)
        ip_lbl = QLabel("Ağ a&dresi")
        ip_lbl.setBuddy(self.ip_combo)
        ip_form.addRow(ip_lbl, self.ip_combo)
        qr_layout.addLayout(ip_form)

        url_row = QHBoxLayout()
        url_row.setSpacing(6)
        self.url_edit = QLineEdit()
        self.url_edit.setReadOnly(True)
        self.copy_btn = QPushButton("Ko&pyala")
        self.copy_btn.setIcon(_theme_icon("edit-copy"))
        self.copy_btn.clicked.connect(self._copy_url)
        self.regen_btn = QPushButton("&Yenile")
        self.regen_btn.setIcon(_theme_icon("view-refresh"))
        self.regen_btn.clicked.connect(self._regenerate_token)

        url_row.addWidget(self.url_edit, 1)
        url_row.addWidget(self.copy_btn)
        url_row.addWidget(self.regen_btn)
        qr_layout.addLayout(url_row)

        steps_lbl = QLabel(
            "1. Telefonla QR kodu okutun (aynı Wi-Fi ağında olmalı).\n"
            "2. Güvenlik uyarısında 'Gelişmiş → Siteye devam et' seçip mikrofon izni verin."
        )
        steps_lbl.setWordWrap(True)
        steps_lbl.setPalette(sub_pal)
        qr_layout.addWidget(steps_lbl)
        qr_layout.addStretch(1)

        # Shortcut & Tab Order
        self._shortcut_rec = QShortcut(QKeySequence("F9"), self)
        self._shortcut_rec.activated.connect(self._toggle_recording)

        self.setTabOrder(self.region_combo, self.pick_region_btn)
        self.setTabOrder(self.pick_region_btn, self.fps_combo)
        self.setTabOrder(self.fps_combo, self.quality_combo)
        self.setTabOrder(self.quality_combo, self.cursor_chk)
        self.setTabOrder(self.cursor_chk, self.system_audio_chk)
        self.setTabOrder(self.system_audio_chk, self.minimize_chk)
        self.setTabOrder(self.minimize_chk, self.record_btn)

    def _make_card(self) -> QFrame:
        card = QFrame(self)
        card.setFrameShape(QFrame.Shape.StyledPanel)
        card.setObjectName("studioCard")
        card.setStyleSheet(
            "#studioCard { border: 1px solid palette(mid); border-radius: 10px; background: palette(base); }"
        )
        return card

    def _section_title(self, text: str) -> QLabel:
        lbl = QLabel(text)
        f = QFont(lbl.font())
        f.setBold(True)
        f.setPointSize(f.pointSize() + 1)
        lbl.setFont(f)
        return lbl

    def _set_badge_style(self, recording: bool) -> None:
        if recording:
            self.state_badge.setStyleSheet(
                "padding: 6px 12px; border-radius: 12px; font-weight: 600;"
                "background: #d70015; color: #ffffff;"
            )
        else:
            self.state_badge.setStyleSheet(
                "padding: 6px 12px; border-radius: 12px; font-weight: 600;"
                "border: 1px solid palette(mid); background: palette(button); color: palette(buttonText);"
            )

    def _apply_record_btn_style(self, recording: bool) -> None:
        if recording:
            self.record_btn.setStyleSheet(
                "QPushButton { background: #d70015; color: #ffffff; border: none; border-radius: 8px; font-weight: 700; padding: 8px 14px; }"
                "QPushButton:hover { background: #b80012; }"
                "QPushButton:pressed { background: #96000e; }"
            )
        else:
            self.record_btn.setStyleSheet(
                "QPushButton { background: #0a84ff; color: #ffffff; border: none; border-radius: 8px; font-weight: 700; padding: 8px 14px; }"
                "QPushButton:hover { background: #0070e0; }"
                "QPushButton:pressed { background: #005bb8; }"
            )

    def _setup_tray(self) -> None:
        self._tray: Optional[QSystemTrayIcon] = None
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        self._tray = QSystemTrayIcon(_make_tray_icon(False), self)
        self._tray.setToolTip("Toshi Ekran Kayıt")
        menu = QMenu(self)
        self._tray_toggle_action = QAction("Kaydı Başlat", self)
        self._tray_toggle_action.triggered.connect(self._toggle_recording)
        show_action = QAction("Pencereyi Göster", self)
        show_action.triggered.connect(self._restore_window)
        quit_action = QAction("Çıkış", self)
        quit_action.triggered.connect(self._quit_app)
        menu.addAction(self._tray_toggle_action)
        menu.addAction(show_action)
        menu.addSeparator()
        menu.addAction(quit_action)
        self._tray.setContextMenu(menu)
        self._tray.activated.connect(self._on_tray_activated)
        self._tray.show()

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self._restore_window()

    def _restore_window(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _format_region_text(self) -> str:
        r = self._selected_region if self._use_custom_region else self._full_region
        return f"Alan: {r.width}×{r.height} (X={r.x}, Y={r.y})"

    def _on_region_mode_changed(self) -> None:
        mode = self.region_combo.currentData()
        self._use_custom_region = mode == "custom"
        if self._use_custom_region and self._selected_region == self._full_region:
            self._open_region_picker()
        else:
            self.region_info_lbl.setText(self._format_region_text())

    def _open_region_picker(self) -> None:
        if self._recorder is not None:
            return
        self.hide()
        QApplication.processEvents()
        time.sleep(0.18)
        overlay = RegionSelectorOverlay()
        overlay.region_selected.connect(self._on_region_picked)
        overlay.cancelled.connect(self._on_region_cancelled)
        self._region_overlay = overlay
        overlay.showFullScreen()

    def _on_region_picked(self, region: CaptureRegion) -> None:
        self._selected_region = region
        self._use_custom_region = True
        self.region_combo.blockSignals(True)
        self.region_combo.setCurrentIndex(1)
        self.region_combo.blockSignals(False)
        self.region_info_lbl.setText(self._format_region_text())
        self._region_overlay = None
        self._restore_window()

    def _on_region_cancelled(self) -> None:
        self._region_overlay = None
        if self._selected_region == self._full_region:
            self._use_custom_region = False
            self.region_combo.blockSignals(True)
            self.region_combo.setCurrentIndex(0)
            self.region_combo.blockSignals(False)
        self.region_info_lbl.setText(self._format_region_text())
        self._restore_window()

    def _refresh_qr(self) -> None:
        ip = self.ip_combo.currentData() or "127.0.0.1"
        url = self._bridge.build_url(str(ip))
        self.url_edit.setText(url)
        try:
            qr_img = generate_qr_image(url, scale=7, border=3)
            pix = _numpy_gray_to_pixmap(qr_img)
            scaled = pix.scaled(
                QSize(210, 210),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
            self.qr_label.setPixmap(scaled)
        except Exception as exc:
            self.qr_label.setText(f"QR oluşturulamadı: {exc}")

    def _copy_url(self) -> None:
        QApplication.clipboard().setText(self.url_edit.text())
        self.copy_btn.setText("Kopyalandı")
        QTimer.singleShot(1400, lambda: self.copy_btn.setText("Ko&pyala"))

    def _regenerate_token(self) -> None:
        self._bridge.regenerate_token()
        self._refresh_qr()

    def _on_phone_pcm(self, pcm_44k: bytes) -> None:
        rec = self._recorder
        if rec is not None and rec.is_running():
            rec.push_pcm(pcm_44k)

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.region_combo.setEnabled(enabled)
        self.pick_region_btn.setEnabled(enabled)
        self.fps_combo.setEnabled(enabled)
        self.quality_combo.setEnabled(enabled)
        self.cursor_chk.setEnabled(enabled)
        self.system_audio_chk.setEnabled(enabled)

    def _toggle_recording(self) -> None:
        if self._recorder is not None:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        region = self._selected_region if self._use_custom_region else self._full_region
        fps = int(self.fps_combo.currentData() or 24)
        quality = str(self.quality_combo.currentData() or "balanced")
        draw_cursor = self.cursor_chk.isChecked()
        sys_source = discover_default_monitor_source() if self.system_audio_chk.isChecked() else None

        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = self.recordings_dir / f"kayit_{stamp}.mp4"

        try:
            self._recorder = ScreenRecorder(
                out_path=out_path,
                region=region,
                fps=fps,
                quality=quality,
                draw_cursor=draw_cursor,
                system_monitor_source=sys_source,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Kayıt Başlatılamadı", f"FFmpeg başlatılamadı:\n{exc}")
            self._recorder = None
            return

        self._recording_started_at = time.monotonic()
        self._set_controls_enabled(False)
        self.record_btn.setText("&Kaydı Durdur (F9)")
        self.record_btn.setIcon(_theme_icon("media-playback-stop"))
        self._apply_record_btn_style(recording=True)
        self.state_badge.setText("Kayıt Sürüyor")
        self._set_badge_style(recording=True)
        self.file_meta_lbl.setText(out_path.name)

        if self._tray is not None:
            self._tray.setIcon(_make_tray_icon(True))
            self._tray_toggle_action.setText("Kaydı Durdur ve Kaydet")

        if self.minimize_chk.isChecked():
            self.showMinimized()

    def _stop_recording(self) -> None:
        rec = self._recorder
        if rec is None:
            return
        self._recorder = None
        self.state_badge.setText("Kaydediliyor…")
        QApplication.processEvents()

        out_path, err_text = rec.stop()
        self._set_controls_enabled(True)
        self.record_btn.setText("&Kaydı Başlat (F9)")
        self.record_btn.setIcon(_theme_icon("media-record"))
        self._apply_record_btn_style(recording=False)
        self.state_badge.setText("Hazır")
        self._set_badge_style(recording=False)

        if self._tray is not None:
            self._tray.setIcon(_make_tray_icon(False))
            self._tray_toggle_action.setText("Kaydı Başlat")

        self._restore_window()

        if out_path.exists() and out_path.stat().st_size > 1024:
            self._last_saved_path = out_path
            self.play_last_btn.setEnabled(True)
            self.file_meta_lbl.setText(f"Kaydedildi: {out_path.name} ({_format_size(out_path)})")
        else:
            self.file_meta_lbl.setText("Kayıt oluşturulamadı")
            if err_text:
                QMessageBox.warning(self, "Kayıt Hatası", f"FFmpeg hata bildirdi:\n{err_text}")

    def _on_tick(self) -> None:
        # Update phone mic indicator & VU meter
        connected = self._bridge.is_client_connected()
        audio_active = self._bridge.is_audio_active()
        raw_lvl = self._bridge.audio_level()
        self._smoothed_level = max(raw_lvl, self._smoothed_level * 0.72)
        self.vu_bar.setValue(int(min(100, round(self._smoothed_level * 100))))

        if connected and audio_active:
            ip = self._bridge.client_info() or "telefon"
            self.mic_dot.setStyleSheet("color: #248a3d; font-size: 16px;")
            self.mic_status_lbl.setText(f"Bağlı ({ip}) — Mikrofon sesi aktarılıyor")
        elif connected:
            ip = self._bridge.client_info() or "telefon"
            self.mic_dot.setStyleSheet("color: #c93400; font-size: 16px;")
            self.mic_status_lbl.setText(f"Bağlı ({ip}) — Telefon sessizde veya izin bekliyor")
        else:
            self.mic_dot.setStyleSheet("color: #b25000; font-size: 16px;")
            self.mic_status_lbl.setText("Telefon bekleniyor — Sağdaki QR kodu taratın")

        # Update active recording timer & watch FFmpeg health
        if self._recorder is not None:
            if not self._recorder.is_running():
                self._stop_recording()
                return
            elapsed = time.monotonic() - self._recording_started_at
            time_str = _format_duration(elapsed)
            self.timer_lbl.setText(time_str)
            self.state_badge.setText(f"Kayıt • {time_str}")
            size_str = _format_size(self._recorder.out_path)
            self.file_meta_lbl.setText(f"{self._recorder.out_path.name} ({size_str})")

    def _open_recordings_dir(self) -> None:
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["xdg-open", str(self.recordings_dir)])

    def _play_last_recording(self) -> None:
        if self._last_saved_path and self._last_saved_path.exists():
            subprocess.Popen(["xdg-open", str(self._last_saved_path)])

    def _quit_app(self) -> None:
        self._force_quit = True
        self.close()

    def closeEvent(self, event) -> None:
        if self._recorder is not None and not self._force_quit:
            ans = QMessageBox.question(
                self,
                "Kayıt Sürüyor",
                "Aktif ekran kaydı durdurulup kaydedilsin mi?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes,
            )
            if ans != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        if self._recorder is not None:
            self._stop_recording()
        self._poll_timer.stop()
        self._bridge.stop()
        if self._tray is not None:
            self._tray.hide()
        event.accept()
