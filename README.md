# Locust Web - Streamlit

Python Streamlit arayüzü ile Locust yük testlerini başlatın ve raporlarını görüntüleyin.

## Kurulum

1) Bağımlılıkları kurun

```
pip install -r requirements.txt
```

2) Uygulamayı başlatın

```
streamlit run app.py
```

3) Tarayıcıda açılan sayfada Locust dosyasını, hedef `host` değerini, kullanıcı sayısı ve çalışma süresi gibi parametreleri girip çalıştırın.

## Çıktılar

- Her çalışma `runs/<timestamp>/` altında saklanır
- CSV istatistikleri: `<prefix>_stats.csv`, `<prefix>_stats_history.csv` vb.
- HTML raporu: `report.html` (seçiliyse)

## Örnek locustfile

`locustfiles/sample_locustfile.py` bir örnek içerir. Kendi dosyalarınızı `locustfiles/` altına ekleyin ve arayüzden seçin.

## Notlar

- HTML raporu için "HTML raporu üret" seçeneğini işaretleyin.
- Zaman serisi grafikleri için "CSV full history" seçeneği açık olmalıdır.

## Eşikler ve Dashboard

- `.env` ile eşik tanımlayabilirsiniz:
  - `THRESHOLD_P95_MS=800` gibi
  - `THRESHOLD_SUCCESS_RATE=99` gibi
- Raporlar sekmesinde koşu özeti altında eşiklere göre uyarı/başarı rozetleri gösterilir.
- Genel Dashboard sekmesinde filtreleme, gruplanmış özet ve grafikler bulunur; eşikler uygulanır.

## Koşu Karşılaştırma

- Raporlar sekmesinde seçili koşuyu başka bir koşu ile kıyaslayabilirsiniz.
