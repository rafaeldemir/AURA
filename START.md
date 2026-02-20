# AURA — Yeni Chat Context

## Ne yapıyoruz
Türkçe prompt'la kıyafet kombin öneren uygulama. Kullanıcı dolabını fotoğrafla yükler, "yarın iş toplantım var" yazar, sistem 3 kombin önerir.

## GitHub
https://github.com/rafaeldemir/AURA ← bunu güncelle

## Tech Stack
- Segmentation: SegFormer b2 (mattmdjaga/segformer_b2_clothes)
- Embedding: CLIP ViT-L/14
- Backend: FastAPI + Uvicorn
- Ortam: Google Colab Pro+ (A100/H100)
- Storage: Şu an in-memory (Supabase V2'de)

## Mevcut Durum
```
Modül 1 — Segmentation       %85 ✓
Modül 2 — Feature Extraction  %90 ✓
Modül 3 — Outfit Engine       %80 ✓
Modül 4 — FastAPI             %80 ✓
Modül 5 — Storage             %0  ← sıradaki
Frontend                      %0  ← sıradaki
```

## Nasıl Başlatılır (Colab)
```python
!git clone https://TOKEN@github.com/KULLANICI/aura.git
!pip install -q git+https://github.com/openai/CLIP.git transformers scipy scikit-learn fastapi uvicorn python-multipart nest_asyncio
import threading
exec(open('aura/aura_core.py').read())
t=threading.Thread(target=run,daemon=True); t.start()
```

## Fotoğraf Yükleme (Colab)
```python
from google.colab import files as colab_files
import requests, shutil

WARDROBE_STORE["test_user"] = []
uploaded = colab_files.upload()

for filename, data in uploaded.items():
    path = f"/content/{filename}"
    with open(path,'wb') as f: f.write(data)
    is_real = not any(x in filename.lower() for x in ["-p.jpg","alternate",".avif"])
    with open(path,'rb') as f:
        r = requests.post("http://localhost:8000/wardrobe/upload",
            params={"user_id":"test_user","real_photo":is_real},
            files={"file":(filename,f,"image/jpeg")})
    result = r.json()
    print(f"\n{filename}:")
    for item in result.get("processed",[]): print(f"  ✓ [{item['category']}] {item['subcategory']} — {item['fabric']}")
    for retry in result.get("retry_needed",[]): print(f"  ↺ {retry['message']}")
```

## Prompt Test
```python
items = WARDROBE_STORE.get("test_user", [])
for prompt in ["yarın iş toplantım var", "konsere gidiyorum", "spor yapmaya gidiyorum"]:
    print(f"\nPrompt: '{prompt}'")
    outfits = generate(items, prompt, current_hour=10)
    for i,o in enumerate(outfits):
        if o.get("message"): print(f"  ⚠ {o['message']}"); break
        print(f"  Outfit {i+1} [{o['style_axis']}] — {o['final_score']}")
        for l,c,f in zip(o["labels"],o["categories"],o["fabrics"]): print(f"    · [{c}] {l} — {f}")
```

## Sistem Nasıl Çalışır
1. Fotoğraf → SegFormer ile kıyafet kesimi
2. Her item için CLIP ile subcategory tespiti (tshirt/shirt/hoodie/knitwear vb)
3. Her item'a formality skoru atanır (0.0-1.0)
4. Prompt → Türkçe keyword matching → occasion (formal/sport/casual vb)
5. Occasion bazlı filtreler uygulanır (ayakkabı, alt giysi, dış giysi)
6. Hava + saat bazlı filtreler (sabah shorts yok, soğukta sandal yok)
7. Renk uyumu (HSV clash detection)
8. CLIP cosine similarity (%65) + formality uyumu (%35) → skor
9. Similarity suppression ile 3 farklı vibe seçilir

## Occasion Sistemi
formal / sport / outdoor / school / event / event_casual / night / casual
Her occasion'ın min formality, bottom tercihi, shoe tercihi var.

## Bilinen Açık Sorunlar
- Formal dolap yetersizse tek outfit geliyor (V1 için normal)
- Modül 5 (Supabase) henüz yok — session kapanınca dolap sıfırlanıyor
- Günlük fotoğraflarda confidence bazen düşük

## Son Oturumdaki Sorun
Son oturum: OOTDiffusion kuruldu, transformers çakışması var, iki notebook çözümü deneyeceğiz
