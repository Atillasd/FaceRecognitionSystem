"""
Yüz Kodlama Modülü
───────────────────
bilinen_yuzler/ klasöründeki resimleri tarar ve OpenFace DNN modeli ile
128 boyutlu face embedding vektörleri üretir.
"""

import os
import cv2
import numpy as np
import tempfile
import shutil


# ── Yollar ───────────────────────────────────────────
PROJE_DIZINI = os.path.dirname(os.path.abspath(__file__))
YUZLER_KLASORU = os.path.join(PROJE_DIZINI, "bilinen_yuzler")
MODELLER_DIZINI = os.path.join(PROJE_DIZINI, "modeller")

EMBEDDING_MODEL = os.path.join(MODELLER_DIZINI, "nn4.small2.v1.t7")
PROTOTXT = os.path.join(MODELLER_DIZINI, "deploy.prototxt")
CAFFEMODEL = os.path.join(MODELLER_DIZINI, "res10_300x300_ssd_iter_140000.caffemodel")

DESTEKLENEN_UZANTILAR = (".jpg", ".jpeg", ".png", ".bmp")

# ── Ayarlar ──────────────────────────────────────────
GUVEN_ESIGI = 0.5


def embedding_modeli_yukle():
    """OpenFace embedding modelini yükler (Türkçe karakter uyumlu)."""
    if not os.path.exists(EMBEDDING_MODEL):
        print(f"❌ Embedding model dosyası bulunamadı: {EMBEDDING_MODEL}")
        return None
    # readNetFromTorch buffer desteklemez, Türkçe karakter içeren yolu
    # da okuyamaz → modeli geçici ASCII-safe bir yola kopyala
    tmp_dir = tempfile.mkdtemp()
    tmp_model = os.path.join(tmp_dir, "model.t7")
    try:
        shutil.copy2(EMBEDDING_MODEL, tmp_model)
        net = cv2.dnn.readNetFromTorch(tmp_model)
        return net
    except Exception as e:
        print(f"❌ Embedding modeli yüklenirken hata: {e}")
        return None
    finally:
        # Geçici dosyayı temizle
        shutil.rmtree(tmp_dir, ignore_errors=True)


def tespit_modeli_yukle():
    """Yüz tespit modelini yükler (Türkçe karakter uyumlu)."""
    if not os.path.exists(PROTOTXT) or not os.path.exists(CAFFEMODEL):
        print("❌ Yüz tespit model dosyaları bulunamadı!")
        return None
    with open(PROTOTXT, 'rb') as f:
        proto_data = f.read()
    with open(CAFFEMODEL, 'rb') as f:
        model_data = f.read()
    proto_buf = np.frombuffer(proto_data, dtype=np.uint8)
    model_buf = np.frombuffer(model_data, dtype=np.uint8)
    return cv2.dnn.readNetFromCaffe(proto_buf, model_buf)


def yuz_embedding_uret(embedder, yuz_resmi):
    """
    Bir yüz resminden 128 boyutlu embedding vektörü üretir.

    Args:
        embedder: OpenFace DNN modeli
        yuz_resmi: BGR formatında yüz resmi (crop edilmiş)

    Returns:
        numpy array (128,) veya None
    """
    try:
        yuz_blob = cv2.dnn.blobFromImage(
            yuz_resmi, 1.0 / 255, (96, 96),
            (0, 0, 0), swapRB=True, crop=False
        )
        embedder.setInput(yuz_blob)
        vec = embedder.forward()
        return vec.flatten()
    except Exception as e:
        print(f"   ❌ Embedding hatası: {e}")
        return None


def _resimden_embedding(embedder, detector, dosya_yolu, dosya_adi):
    """Tek bir resim dosyasından embedding üretir. Başarısızsa None döner."""
    try:
        buf = np.fromfile(dosya_yolu, dtype=np.uint8)
        resim = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if resim is None:
            print(f"   ⚠️  {dosya_adi} → Resim okunamadı, atlanıyor")
            return None

        (h, w) = resim.shape[:2]
        yuz_resmi = resim

        if detector is not None and min(h, w) > 100:
            blob = cv2.dnn.blobFromImage(
                cv2.resize(resim, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0)
            )
            detector.setInput(blob)
            tespitler = detector.forward()

            for i in range(tespitler.shape[2]):
                guven = tespitler[0, 0, i, 2]
                if guven > GUVEN_ESIGI:
                    kutu = tespitler[0, 0, i, 3:7] * [w, h, w, h]
                    (sol, ust, sag, alt) = kutu.astype("int")
                    sol, ust = max(0, sol), max(0, ust)
                    sag, alt = min(w, sag), min(h, alt)
                    yuz_resmi = resim[ust:alt, sol:sag]
                    break

        embedding = yuz_embedding_uret(embedder, yuz_resmi)
        if embedding is not None:
            print(f"   ✅ {dosya_adi} → yüklendi")
        else:
            print(f"   ⚠️  {dosya_adi} → Embedding üretilemedi, atlanıyor")
        return embedding

    except Exception as e:
        print(f"   ❌ {dosya_adi} → Hata: {e}")
        return None


def yuzleri_yukle():
    """
    bilinen_yuzler/ klasöründeki kişi alt klasörlerini ve düz dosyaları tarar,
    her kişi için embedding'lerin ortalamasını alarak daha güvenilir tanıma sağlar.

    Yapı:
        bilinen_yuzler/
        ├── Atilla/        ← klasör adı = kişi adı
        │   ├── 1.jpg
        │   └── 2.jpg
        └── eski_foto.jpg  ← geriye dönük uyumluluk (dosya adı = kişi adı)

    Returns:
        tuple: (bilinen_encodingler, bilinen_isimler)
    """
    bilinen_encodingler = []
    bilinen_isimler = []

    if not os.path.exists(YUZLER_KLASORU):
        print("⚠️  bilinen_yuzler/ klasörü bulunamadı!")
        return bilinen_encodingler, bilinen_isimler

    # Modelleri yükle
    embedder = embedding_modeli_yukle()
    detector = tespit_modeli_yukle()
    if embedder is None:
        return bilinen_encodingler, bilinen_isimler

    # ── 1) Kişi alt klasörlerini tara ──
    toplam_foto = 0
    for girdi in sorted(os.listdir(YUZLER_KLASORU)):
        kisi_yolu = os.path.join(YUZLER_KLASORU, girdi)
        if not os.path.isdir(kisi_yolu):
            continue

        isim = girdi
        dosyalar = [
            f for f in os.listdir(kisi_yolu)
            if f.lower().endswith(DESTEKLENEN_UZANTILAR)
        ]
        if not dosyalar:
            continue

        print(f"\n🔄 '{isim}' — {len(dosyalar)} fotoğraf yükleniyor...")
        toplam_foto += len(dosyalar)

        kisi_embeddingleri = []
        for dosya in sorted(dosyalar):
            dosya_yolu = os.path.join(kisi_yolu, dosya)
            emb = _resimden_embedding(embedder, detector, dosya_yolu, f"{isim}/{dosya}")
            if emb is not None:
                kisi_embeddingleri.append(emb)

        if kisi_embeddingleri:
            # Tüm embedding'lerin ortalamasını al → daha güvenilir tanıma
            ortalama = np.mean(kisi_embeddingleri, axis=0)
            # Normalize et
            ortalama = ortalama / np.linalg.norm(ortalama)
            bilinen_encodingler.append(ortalama)
            bilinen_isimler.append(isim)
            print(f"   📊 '{isim}' için {len(kisi_embeddingleri)} embedding ortalaması alındı")

    # ── 2) Geriye dönük uyumluluk: düz dosyalar ──
    duz_dosyalar = [
        f for f in os.listdir(YUZLER_KLASORU)
        if os.path.isfile(os.path.join(YUZLER_KLASORU, f))
        and f.lower().endswith(DESTEKLENEN_UZANTILAR)
    ]
    if duz_dosyalar:
        print(f"\n🔄 {len(duz_dosyalar)} düz dosya yükleniyor (eski format)...")
        toplam_foto += len(duz_dosyalar)

        # Düz dosyaları kişilere göre grupla
        kisi_gruplari = {}
        for dosya in duz_dosyalar:
            isim = os.path.splitext(dosya)[0]
            parcalar = isim.rsplit("_", 1)
            if len(parcalar) == 2 and parcalar[1].isdigit():
                isim = parcalar[0]
            if isim not in kisi_gruplari:
                kisi_gruplari[isim] = []
            kisi_gruplari[isim].append(dosya)

        for isim, dosyalar in kisi_gruplari.items():
            # Eğer klasör tabanlı zaten yüklendiyse atla
            if isim in bilinen_isimler:
                print(f"   ⏭️  '{isim}' zaten klasörden yüklendi, düz dosyalar atlanıyor")
                continue

            kisi_embeddingleri = []
            for dosya in dosyalar:
                dosya_yolu = os.path.join(YUZLER_KLASORU, dosya)
                emb = _resimden_embedding(embedder, detector, dosya_yolu, dosya)
                if emb is not None:
                    kisi_embeddingleri.append(emb)

            if kisi_embeddingleri:
                ortalama = np.mean(kisi_embeddingleri, axis=0)
                ortalama = ortalama / np.linalg.norm(ortalama)
                bilinen_encodingler.append(ortalama)
                bilinen_isimler.append(isim)
                print(f"   📊 '{isim}' için {len(kisi_embeddingleri)} embedding ortalaması alındı")

    print(f"\n✅ Toplam {toplam_foto} fotoğraftan {len(bilinen_encodingler)} kişi yüklendi.\n")
    return bilinen_encodingler, bilinen_isimler


if __name__ == "__main__":
    encodingler, isimler = yuzleri_yukle()
    if isimler:
        print("Kayıtlı kişiler:", ", ".join(set(isimler)))
