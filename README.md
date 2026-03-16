# Yüz Tanıma Sistemi

Bu proje, Python kullanarak yüz tanıma ve kayıt işlemlerini gerçekleştiren basit bir sistemdir. OpenCV ve derin öğrenme modelleri kullanarak yüzleri algılar, kodlar ve tanır.

## Özellikler

- Yüz kayıt etme
- Yüz kodlama (embedding)
- Gerçek zamanlı yüz tanıma
- Bilinen yüzler veritabanı

## Gereksinimler

- Python 3.7+
- OpenCV
- NumPy
- Caffe modelleri (modeller/ klasöründe mevcut)

## Kurulum

1. Gerekli paketleri yükleyin:
   ```
   pip install opencv-python numpy
   ```

2. Projeyi klonlayın veya indirin.

3. Modelleri `modeller/` klasörüne yerleştirin (zaten mevcut).

## Kullanım

### Yüz Kayıt
```
python yuz_kayit.py
```

### Yüz Kodlama
```
python yuz_kodlama.py
```

### Yüz Tanıma
```
python yuz_tanima.py
```

### Ana Uygulama
```
python main.py
```

## Dosya Yapısı

- `main.py`: Ana uygulama dosyası
- `yuz_kayit.py`: Yüz kayıt modülü
- `yuz_kodlama.py`: Yüz kodlama modülü
- `yuz_tanima.py`: Yüz tanıma modülü
- `bilinen_yuzler/`: Kayıtlı yüzler (fotoğraflar)
- `modeller/`: Derin öğrenme modelleri

## Notlar

- `__pycache__/` klasörü Git'te yok sayılır.
- Kişisel veriler için `bilinen_yuzler/` klasörünü dikkatli kullanın.

## Lisans

Bu proje açık kaynaklıdır. Dilediğiniz gibi kullanabilirsiniz.