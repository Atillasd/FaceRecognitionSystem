"""
Yüz Tanıma Sistemi — Ana Menü
══════════════════════════════
Faz 1: Yerel kamera ile yüz kayıt ve gerçek zamanlı tanıma.
"""

import os
from yuz_kayit import yuz_kaydet
from yuz_tanima import tanima_baslat


# ── Yollar ───────────────────────────────────────────
PROJE_DIZINI = os.path.dirname(os.path.abspath(__file__))
YUZLER_KLASORU = os.path.join(PROJE_DIZINI, "bilinen_yuzler")
DESTEKLENEN_UZANTILAR = (".jpg", ".jpeg", ".png", ".bmp")


BANNER = """
╔══════════════════════════════════════════╗
║       🔍 YÜZ TANIMA SİSTEMİ v2.0       ║
║──────────────────────────────────────────║
║  Faz 1 — Yerel Kamera                   ║
╚══════════════════════════════════════════╝
"""


def kayitli_kisileri_listele():
    """Kayıtlı kişileri ve fotoğraf sayılarını gösterir."""
    if not os.path.exists(YUZLER_KLASORU):
        print("\n  ⚠️  Henüz kayıtlı kişi yok!")
        return

    kisiler = []
    for girdi in sorted(os.listdir(YUZLER_KLASORU)):
        kisi_yolu = os.path.join(YUZLER_KLASORU, girdi)
        if os.path.isdir(kisi_yolu):
            foto_sayisi = len([
                f for f in os.listdir(kisi_yolu)
                if f.lower().endswith(DESTEKLENEN_UZANTILAR)
            ])
            if foto_sayisi > 0:
                kisiler.append((girdi, foto_sayisi))

    if not kisiler:
        print("\n  ⚠️  Henüz kayıtlı kişi yok!")
        return

    print("\n  ╔══════════════════════════════════╗")
    print("  ║      📋 Kayıtlı Kişiler         ║")
    print("  ╠══════════════════════════════════╣")
    for isim, sayi in kisiler:
        print(f"  ║  👤 {isim:<15} 📸 {sayi} foto  ║")
    print("  ╚══════════════════════════════════╝")
    print(f"\n  Toplam: {len(kisiler)} kişi\n")


def menu_goster():
    """Ana menüyü gösterir."""
    print(BANNER)
    print("  [1] 📸 Yeni Yüz Kaydet")
    print("  [2] 🎯 Yüz Tanımayı Başlat")
    print("  [3] 📋 Kayıtlı Kişileri Listele")
    print("  [4] 🚪 Çıkış")
    print()


def main():
    while True:
        menu_goster()
        secim = input("  Seçiminiz (1/2/3/4): ").strip()

        if secim == "1":
            yuz_kaydet()

        elif secim == "2":
            tanima_baslat()

        elif secim == "3":
            kayitli_kisileri_listele()

        elif secim == "4":
            print("\n  👋 Güle güle!\n")
            break

        else:
            print("\n  ❌ Geçersiz seçim! Lütfen 1, 2, 3 veya 4 girin.\n")


if __name__ == "__main__":
    main()

