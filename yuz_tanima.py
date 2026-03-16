"""
Gerçek Zamanlı Yüz Tanıma Modülü
─────────────────────────────────
Kameradan sürekli görüntü alarak kayıtlı yüzlerle karşılaştırır.
OpenCV DNN (ResNet SSD + OpenFace) kullanır.
"""

import cv2
import numpy as np
import os
from yuz_kodlama import yuzleri_yukle, yuz_embedding_uret, embedding_modeli_yukle


# ── Yollar ───────────────────────────────────────────
PROJE_DIZINI = os.path.dirname(os.path.abspath(__file__))
MODELLER_DIZINI = os.path.join(PROJE_DIZINI, "modeller")
PROTOTXT = os.path.join(MODELLER_DIZINI, "deploy.prototxt")
CAFFEMODEL = os.path.join(MODELLER_DIZINI, "res10_300x300_ssd_iter_140000.caffemodel")

# ── Ayarlar ──────────────────────────────────────────
GUVEN_ESIGI = 0.5          # Yüz tespiti minimum güven
ESLESME_ESIGI = 0.7        # Embedding mesafe eşiği (düşük = daha katı)
FRAME_ATLAMA = 2           # Her N frame'de bir tanıma yap (performans)
KUTU_RENGI = (0, 255, 0)         # Tanınan yüz — yeşil
BILINMEYEN_RENGI = (0, 0, 255)   # Bilinmeyen yüz — kırmızı


def tanima_baslat():
    """Gerçek zamanlı yüz tanıma döngüsünü başlatır."""

    # Kayıtlı yüzleri yükle
    bilinen_encodingler, bilinen_isimler = yuzleri_yukle()
    if not bilinen_encodingler:
        print("❌ Hiç kayıtlı yüz bulunamadı! Önce yüz kaydedin.")
        return

    # Modelleri yükle
    print("🔄 Modeller yükleniyor...")

    # Türkçe karakter uyumlu model yükleme
    with open(PROTOTXT, 'rb') as f:
        proto_data = f.read()
    with open(CAFFEMODEL, 'rb') as f:
        model_data = f.read()
    proto_buf = np.frombuffer(proto_data, dtype=np.uint8)
    model_buf = np.frombuffer(model_data, dtype=np.uint8)
    detector = cv2.dnn.readNetFromCaffe(proto_buf, model_buf)

    embedder = embedding_modeli_yukle()

    if embedder is None:
        print("❌ Embedding modeli yüklenemedi!")
        return

    print("📹 Kamera açılıyor... (Q ile çıkış)\n")
    kamera = cv2.VideoCapture(0)

    if not kamera.isOpened():
        print("❌ Kamera açılamadı!")
        return

    kamera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_sayaci = 0
    sonuclar = []   # [(sol, ust, sag, alt, isim, mesafe), ...]

    while True:
        ret, frame = kamera.read()
        if not ret:
            print("❌ Kameradan görüntü alınamadı!")
            break

        frame_sayaci += 1
        (yukseklik, genislik) = frame.shape[:2]

        # Performans: her N frame'de bir tanıma yap
        if frame_sayaci % FRAME_ATLAMA == 0:
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0)
            )
            detector.setInput(blob)
            tespitler = detector.forward()

            yeni_sonuclar = []

            for i in range(tespitler.shape[2]):
                guven = tespitler[0, 0, i, 2]
                if guven < GUVEN_ESIGI:
                    continue

                kutu = tespitler[0, 0, i, 3:7] * [genislik, yukseklik, genislik, yukseklik]
                (sol, ust, sag, alt) = kutu.astype("int")
                sol = max(0, sol)
                ust = max(0, ust)
                sag = min(genislik, sag)
                alt = min(yukseklik, alt)

                # Yüz bölgesini kes
                yuz = frame[ust:alt, sol:sag]
                if yuz.shape[0] < 20 or yuz.shape[1] < 20:
                    continue

                # Embedding üret
                embedding = yuz_embedding_uret(embedder, yuz)
                if embedding is None:
                    continue

                # Kayıtlı yüzlerle karşılaştır
                isim = "Bilinmeyen"
                en_kucuk_mesafe = float('inf')

                for j, bilinen_enc in enumerate(bilinen_encodingler):
                    mesafe = np.linalg.norm(bilinen_enc - embedding)
                    if mesafe < en_kucuk_mesafe:
                        en_kucuk_mesafe = mesafe
                        if mesafe < ESLESME_ESIGI:
                            isim = bilinen_isimler[j]

                yeni_sonuclar.append((sol, ust, sag, alt, isim, en_kucuk_mesafe))

            sonuclar = yeni_sonuclar

        # Sonuçları frame üzerine çiz
        for (sol, ust, sag, alt, isim, mesafe) in sonuclar:
            renk = KUTU_RENGI if isim != "Bilinmeyen" else BILINMEYEN_RENGI

            # Yüz çerçevesi
            cv2.rectangle(frame, (sol, ust), (sag, alt), renk, 2)

            # İsim arka plan kutusu
            cv2.rectangle(frame, (sol, alt), (sag, alt + 30), renk, cv2.FILLED)

            # Güven yüzdesi (mesafe 0 = mükemmel eşleşme)
            guven_yuzde = max(0, (1 - mesafe)) * 100
            if isim != "Bilinmeyen":
                etiket = f"{isim} (%{guven_yuzde:.0f})"
            else:
                etiket = "Bilinmeyen"

            cv2.putText(frame, etiket, (sol + 6, alt + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Bilgi çubuğu
        cv2.putText(frame, f"Tespit: {len(sonuclar)} yuz | Q: Cikis",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Kayıtlı kişiler
        kayitli = ", ".join(set(bilinen_isimler))
        cv2.putText(frame, f"Kayitli: {kayitli}",
                    (10, yukseklik - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1)

        cv2.imshow("Yuz Tanima Sistemi", frame)

        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            break

    kamera.release()
    cv2.destroyAllWindows()
    print("\n🛑 Yüz tanıma durduruldu.")


if __name__ == "__main__":
    tanima_baslat()
