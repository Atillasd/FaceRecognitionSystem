"""
Yüz Kayıt Modülü
─────────────────
Kameradan yüz fotoğrafı çekerek bilinen_yuzler/ klasörüne kaydeder.
OpenCV DNN ile yüz tespiti yapar.
"""

import cv2
import numpy as np
import os


# ── Yollar ───────────────────────────────────────────
PROJE_DIZINI = os.path.dirname(os.path.abspath(__file__))
YUZLER_KLASORU = os.path.join(PROJE_DIZINI, "bilinen_yuzler")
MODELLER_DIZINI = os.path.join(PROJE_DIZINI, "modeller")

PROTOTXT = os.path.join(MODELLER_DIZINI, "deploy.prototxt")
CAFFEMODEL = os.path.join(MODELLER_DIZINI, "res10_300x300_ssd_iter_140000.caffemodel")

# ── Ayarlar ──────────────────────────────────────────
GUVEN_ESIGI = 0.5   # Yüz tespiti güven eşiği
DESTEKLENEN_UZANTILAR = (".jpg", ".jpeg", ".png", ".bmp")


def klasoru_olustur(kisi_adi=None):
    """bilinen_yuzler klasörünü ve kişi alt klasörünü yoksa oluşturur."""
    if not os.path.exists(YUZLER_KLASORU):
        os.makedirs(YUZLER_KLASORU)
        print(f"[+] '{YUZLER_KLASORU}' klasörü oluşturuldu.")
    if kisi_adi:
        kisi_klasoru = os.path.join(YUZLER_KLASORU, kisi_adi)
        if not os.path.exists(kisi_klasoru):
            os.makedirs(kisi_klasoru)
            print(f"[+] '{kisi_adi}' klasörü oluşturuldu.")
        return kisi_klasoru
    return YUZLER_KLASORU


def yuz_tespit_modeli_yukle():
    """OpenCV DNN yüz tespit modelini yükler.
    Türkçe karakter içeren dosya yolları için dosyayı byte olarak okur."""
    if not os.path.exists(PROTOTXT) or not os.path.exists(CAFFEMODEL):
        print("❌ Model dosyaları bulunamadı!")
        print(f"   Gerekli: {PROTOTXT}")
        print(f"   Gerekli: {CAFFEMODEL}")
        return None

    # OpenCV DNN Türkçe karakter içeren yolları okuyamıyor,
    # bu yüzden dosyaları Python ile okuyup byte buffer'dan yüklüyoruz
    with open(PROTOTXT, 'rb') as f:
        proto_data = f.read()
    with open(CAFFEMODEL, 'rb') as f:
        model_data = f.read()

    proto_buf = np.frombuffer(proto_data, dtype=np.uint8)
    model_buf = np.frombuffer(model_data, dtype=np.uint8)

    return cv2.dnn.readNetFromCaffe(proto_buf, model_buf)


def yuzleri_bul(net, frame, genislik, yukseklik):
    """Frame içindeki yüzleri tespit eder. Koordinat listesi döndürür."""
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    tespitler = net.forward()

    yuzler = []
    for i in range(tespitler.shape[2]):
        guven = tespitler[0, 0, i, 2]
        if guven > GUVEN_ESIGI:
            kutu = tespitler[0, 0, i, 3:7] * [genislik, yukseklik, genislik, yukseklik]
            (sol, ust, sag, alt) = kutu.astype("int")
            # Sınırları kontrol et
            sol = max(0, sol)
            ust = max(0, ust)
            sag = min(genislik, sag)
            alt = min(yukseklik, alt)
            yuzler.append((sol, ust, sag, alt, float(guven)))
    return yuzler


def sonraki_foto_numarasi(kisi_klasoru):
    """Kişi klasöründeki mevcut fotoğraflardan sonraki numarayı döndürür."""
    mevcut = [f for f in os.listdir(kisi_klasoru) if f.lower().endswith(DESTEKLENEN_UZANTILAR)]
    if not mevcut:
        return 1
    numaralar = []
    for f in mevcut:
        ad = os.path.splitext(f)[0]
        if ad.isdigit():
            numaralar.append(int(ad))
    return max(numaralar, default=0) + 1


def yuz_kaydet():
    """Kamerayı açar, kullanıcının yüz fotoğrafını çeker ve kişi klasörüne kaydeder."""

    # Modeli yükle
    net = yuz_tespit_modeli_yukle()
    if net is None:
        return

    isim = input("\n📝 Kayıt edilecek kişinin adını girin: ").strip()
    if not isim:
        print("❌ İsim boş olamaz!")
        return

    # Kişi klasörünü oluştur
    kisi_klasoru = klasoru_olustur(isim)
    mevcut_sayi = len([f for f in os.listdir(kisi_klasoru) if f.lower().endswith(DESTEKLENEN_UZANTILAR)])
    print(f"\n📂 '{isim}' klasöründe {mevcut_sayi} mevcut fotoğraf var.")

    print("\n📸 Kamera açılıyor...")
    print("   → SPACE tuşuna basarak fotoğraf çekin")
    print("   → Q tuşuna basarak çıkın\n")

    kamera = cv2.VideoCapture(0)
    if not kamera.isOpened():
        print("❌ Kamera açılamadı!")
        return

    kamera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    foto_sayaci = 0
    siradaki_numara = sonraki_foto_numarasi(kisi_klasoru)

    while True:
        ret, frame = kamera.read()
        if not ret:
            print("❌ Kameradan görüntü alınamadı!")
            break

        (yukseklik, genislik) = frame.shape[:2]
        yuzler = yuzleri_bul(net, frame, genislik, yukseklik)

        # Yüzleri çiz
        for (sol, ust, sag, alt, guven) in yuzler:
            cv2.rectangle(frame, (sol, ust), (sag, alt), (0, 255, 0), 2)
            etiket = f"Yuz (%{guven * 100:.0f})"
            cv2.putText(frame, etiket, (sol, ust - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Bilgi metinleri
        cv2.putText(frame, f"Kisi: {isim}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Toplam: {mevcut_sayi + foto_sayaci} | Yeni: {foto_sayaci}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(frame, "SPACE: Fotograf Cek | Q: Cikis", (10, yukseklik - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Yuz Kayit - Yuz Tanima Sistemi", frame)

        tus = cv2.waitKey(1) & 0xFF

        if tus == ord(' '):  # SPACE — fotoğraf çek
            if not yuzler:
                print("⚠️  Yüz tespit edilemedi, tekrar deneyin!")
            else:
                # Sadece yüz bölgesini kaydet (daha iyi encoding için)
                (sol, ust, sag, alt, _) = yuzler[0]
                yuz_resmi = frame[ust:alt, sol:sag]
                dosya_adi = f"{siradaki_numara}.jpg"
                dosya_yolu = os.path.join(kisi_klasoru, dosya_adi)
                # cv2.imwrite Türkçe karakter içeren yollarla çalışamıyor
                # Bu yüzden imencode + Python open kullanıyoruz
                basarili, buffer = cv2.imencode('.jpg', yuz_resmi)
                if basarili:
                    with open(dosya_yolu, 'wb') as f:
                        f.write(buffer.tobytes())
                    foto_sayaci += 1
                    siradaki_numara += 1
                    print(f"✅ Fotoğraf kaydedildi: {dosya_yolu}")
                else:
                    print("❌ Fotoğraf encode edilemedi!")

        elif tus == ord('q') or tus == ord('Q'):
            break

    kamera.release()
    cv2.destroyAllWindows()

    if foto_sayaci > 0:
        print(f"\n✅ '{isim}' için {foto_sayaci} yeni fotoğraf kaydedildi (toplam: {mevcut_sayi + foto_sayaci}).")
    else:
        print("\n⚠️  Hiç fotoğraf kaydedilmedi.")


if __name__ == "__main__":
    yuz_kaydet()
