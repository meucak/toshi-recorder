# Toshi Ekran Kayıt (`toshi-recorder`)

Toshiba Satellite L735 (Debian LXQt / X11) için PyQt6 masaüstü ekran kaydedici ve uygulamasız QR kodlu telefon mikrofon köprüsü.

## Özellikler

- **Uygulamasız Telefon Mikrofonu (QR Kod + HTTPS/WSS):** Telefon tarayıcısından (`Chrome` / `Safari`) QR kod ile yerel ağa bağlanır; bilgisayarda çalışan bir mikrofon olmasa dahi sesi gerçek zamanlı olarak ekran kaydına ekler.
- **Tam Ekran veya Özel Bölge Seçimi:** Tek tıkla tam ekran ya da compositor gerektirmeyen sürükle-bırak ekran bölgesi seçimi.
- **Opsiyonel Sistem Sesi Karıştırma:** Telefon mikrofonuna ek olarak bilgisayarın hoparlör çıkışını (`PulseAudio monitor`) da kayda dahil edebilme.
- **Canlı Ses Seviyesi (VU Metre) & Zamanlayıcı:** Telefon bağlantı durumu, canlı mikrofon seviyesi, `F9` hızlı başlat/durdur kısayolu ve sistem tepsisi desteği.

## Kurulum & Çalıştırma

### Gereksinimler (Debian / Ubuntu)
```bash
sudo apt install python3-pyqt6 python3-opencv python3-numpy python3-websockets pulseaudio-utils openssl ffmpeg
```

### Çalıştırma
Kurulu kısayollar ile:
```bash
recscreen
# veya:
toshi-recorder
ekran
```

Depo dizininden doğrudan çalıştırmak için:
```bash
PYTHONPATH=src python3 -m toshi_recorder.main
```

Kayıtlar varsayılan olarak `~/EkranKayitlari` dizinine `.mp4` (`H.264 + AAC`) olarak kaydedilir (`recscreen --saves` ile klasörü açabilirsiniz).

### Testler
```bash
PYTHONPATH=src pytest -v
```
