# AURA — Yeni Chat Context

## Ne yapıyoruz
Türkçe prompt'la kıyafet kombin öneren mobil uygulama.
Kullanıcı dolabını fotoğrafla yükler → "yarın iş toplantım var" yazar → sistem 3 kombin önerir → avatar üzerinde giydirme gösterir.

## GitHub
https://github.com/rafaeldemir/AURA (private)

## Tech Stack
- Segmentation: SegFormer b2 (mattmdjaga/segformer_b2_clothes)
- Embedding: CLIP ViT-L/14
- Backend: FastAPI + Uvicorn
- Storage: Supabase (PostgreSQL + Storage) ✓ ÇALIŞIYOR
- Giydirme: IDM-VTON HuggingFace API
- Ortam: Google Colab Pro+ (A100/H100)
- Frontend: React Native (henüz başlanmadı)

## Nasıl Başlatılır (Colab)
TOKEN = "tokenin"
USER  = "rafaeldemir"
!git clone https://{TOKEN}@github.com/{USER}/AURA.git /content/aura 2>/dev/null || git -C /content/aura pull
!pip install -q git+https://github.com/openai/CLIP.git transformers scipy scikit-learn fastapi uvicorn python-multipart nest_asyncio supabase
import threading
exec(open('/content/aura/aura_core.py').read())
t = threading.Thread(target=run, daemon=True)
t.start()

## Mevcut Durum
Modül 1 — Segmentation          %85 ✓
Modül 2 — Feature Extraction     %90 ✓
Modül 3 — Outfit Engine          %82 ✓
Modül 4 — FastAPI                %80 ✓
Modül 5 — Supabase Storage       %70 ✓ (çalışıyor, CLIP embedding eksik)
Modül 6 — Avatar Sistemi         %80 ✓ (10 avatar hazır, render çalışıyor)
Modül 7 — IDM-VTON Giydirme     %25 ✗ (API çalışıyor, avatar entegrasyonu yok)
Modül 8 — Ten Rengi Tespiti      %0  ✗
Frontend (React Native)          %0  ✗
Genel: %65

## Supabase Bilgileri
- URL: https://xzobaigtiueobqkielnn.supabase.co
- Tablolar: users, wardrobe_items, outfit_feedback
- Storage bucket: wardrobe (public)
- Test user_id: da913b18-8cd8-4b43-aba4-256b870b353d
- RLS: storage.objects için allow_all policy var

## Avatar Sistemi
- 10 avatar: avatar_m/f_fair/light/medium/brown/dark.png
- Klasör: /content/aura/avatars/
- Erkek: atlet + şort, Kadın: sports bra + şort
- Yüz blurlu, beyaz arka plan
- avatar_renderer.py ile render ediliyor

## IDM-VTON
- HuggingFace Space API ile çalışıyor (gradio_client)
- Avatar üzerine giydirme henüz entegre değil

## Outfit Engine
- 8 occasion sistemi (formal/sport/outdoor/school/event/event_casual/night/casual)
- Türkçe keyword parsing
- Renk scoring, formality gap filter, weighted random diversity
- Similarity suppression, güneş boost

## Onboarding Felsefesi
- Selfie → ten rengi otomatik tespit → en yakın avatar seç
- 5 kıyafet yükle → ilk kombin → WOW anı
- Hava durumu izni SONRA iste
- 10 dakikadan kısa onboarding hedefi

## Sıradaki Adımlar
1. CLIP embedding pgvector
2. Gerçek kıyafet tam pipeline test
3. Ten rengi tespiti (selfie oval maske)
4. IDM-VTON + avatar entegrasyonu
5. React Native başlangıç

## Son Oturum
- Supabase kuruldu, tablolar oluşturuldu
- Storage bucket çalışıyor, görsel yükleme OK
- Session kapatıp açınca veri Supabase'den geliyor ✓
