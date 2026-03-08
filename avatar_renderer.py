import cv2, numpy as np, os
from PIL import Image, ImageFilter

# ── ARKA PLAN ─────────────────────────────────────────────
def create_background(w, h, color=None):
    """
    color: None → koyu gri (default)
           "white" → beyaz
           "black" → siyah
           "#RRGGBB" → hex renk
    """
    if color is None or color == "dark":
        base = np.array([0x26, 0x25, 0x24], dtype=np.float32)
        center = np.array([0x2E, 0x2C, 0x2B], dtype=np.float32)
    elif color == "white":
        base = np.array([245, 245, 245], dtype=np.float32)
        center = np.array([255, 255, 255], dtype=np.float32)
    elif color == "black":
        base = np.array([10, 10, 10], dtype=np.float32)
        center = np.array([25, 25, 25], dtype=np.float32)
    elif color.startswith("#"):
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        base = np.array([b, g, r], dtype=np.float32)
        center = np.clip(base * 1.1, 0, 255)
    else:
        base = np.array([0x26, 0x25, 0x24], dtype=np.float32)
        center = np.array([0x2E, 0x2C, 0x2B], dtype=np.float32)

    bg = np.full((h, w, 3), base, dtype=np.float32)
    cx, cy = w // 2, h // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X-cx)**2 + (Y-cy)**2)
    max_dist = np.sqrt(cx**2 + cy**2)
    t = np.clip(dist / (max_dist * 0.65), 0, 1)[:,:,np.newaxis]
    bg = center * (1-t) + base * t
    noise = np.random.normal(0, 6, (h, w)).astype(np.float32)
    bg = np.clip(bg + noise[:,:,np.newaxis] * 0.025, 0, 255)
    vign = 1.0 - np.clip(dist / max_dist, 0, 1) * 0.15
    return np.clip(bg * vign[:,:,np.newaxis], 0, 255).astype(np.uint8)

# ── YÜZ BLUR ──────────────────────────────────────────────
def apply_face_blur(arr):
    """
    MediaPipe ile yüz tespit et, oval gaussian blur uygula.
    Tespit edilemezse üst %22'yi blur yap.
    """
    h, w = arr.shape[:2]
    try:
        import mediapipe as mp
        mp_face = mp.solutions.face_detection
        with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.3) as fd:
            fd_res = fd.process(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        if fd_res and fd_res.detections:
            bb = fd_res.detections[0].location_data.relative_bounding_box
            cx = int((bb.xmin + bb.width/2) * w)
            cy = int((bb.ymin + bb.height/2) * h)
            rx = int(bb.width * w * 0.52)
            ry = int(bb.height * h * 0.56)
        else:
            cx, cy = w//2, int(h*0.13)
            rx, ry = int(w*0.10), int(h*0.09)
    except:
        cx, cy = w//2, int(h*0.13)
        rx, ry = int(w*0.10), int(h*0.09)

    face_blurred = cv2.GaussianBlur(arr, (151, 151), 0)
    fmask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(fmask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
    fmask = cv2.GaussianBlur(fmask, (0, 0), rx * 0.6)
    result = arr.copy()
    for c in range(3):
        result[:,:,c] = (arr[:,:,c]*(1-fmask) + face_blurred[:,:,c]*fmask).astype(np.uint8)
    return result

# ── TEN RENGİ — REINHARD COLOR TRANSFER ───────────────────
def reinhard_skin_transfer(avatar_arr, target_hex):
    """
    Reinhard Color Transfer ile ten rengi değiştir.
    Pastel görünmez — gerçekçi sonuç verir.
    target_hex: "#C68642" gibi hex renk kodu
    """
    import colorsys
    r = int(target_hex[1:3], 16) / 255.0
    g = int(target_hex[3:5], 16) / 255.0
    b = int(target_hex[5:7], 16) / 255.0

    # Hedef rengi LAB'a çevir
    target_rgb = np.array([[[r*255, g*255, b*255]]], dtype=np.uint8)
    target_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)[0,0]

    # Avatar'ı LAB'a çevir
    lab = cv2.cvtColor(avatar_arr, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Ten maskesi — HSV ile tespit
    hsv = cv2.cvtColor(avatar_arr, cv2.COLOR_RGB2HSV)
    skin_mask = cv2.inRange(hsv,
        np.array([0, 20, 70], dtype=np.uint8),
        np.array([25, 200, 255], dtype=np.uint8))
    skin_mask = cv2.GaussianBlur(skin_mask.astype(np.float32)/255, (21,21), 0)

    # Sadece ten bölgesinde renk transferi
    result_lab = lab.copy()
    result_lab[:,:,0] = lab[:,:,0] * (1-skin_mask*0.3) + target_lab[0] * (skin_mask*0.3)
    result_lab[:,:,1] = lab[:,:,1] * (1-skin_mask*0.7) + target_lab[1] * (skin_mask*0.7)
    result_lab[:,:,2] = lab[:,:,2] * (1-skin_mask*0.7) + target_lab[2] * (skin_mask*0.7)

    result = cv2.cvtColor(np.clip(result_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return result

# ── AVATAR SEÇ ────────────────────────────────────────────
def select_avatar(height_cm, weight_kg, gender="male"):
    """
    Boy ve kiloya göre avatar seç.
    """
    # Boy sınıfı
    if height_cm < 168: height_class = "short"
    elif height_cm <= 180: height_class = "medium"
    else: height_class = "tall"

    # BMI bazlı vücut tipi
    bmi = weight_kg / ((height_cm/100)**2)
    if bmi < 18.5: body_class = "slim"
    elif bmi < 25: body_class = "normal"
    elif bmi < 30: body_class = "athletic"
    else: body_class = "plus"

    prefix = "avatar_female_" if gender == "female" else "avatar_"
    filename = f"{prefix}{body_class}_{height_class}.png"
    avatar_path = f"/content/aura/avatars/{filename}"

    # Fallback
    if not os.path.exists(avatar_path):
        fallback = "avatar_female_normal_medium.png" if gender=="female" else "avatar_normal_medium.png"
        avatar_path = f"/content/aura/avatars/{fallback}"

    print(f"✓ Avatar: {body_class}/{height_class} → {os.path.basename(avatar_path)}")
    return avatar_path, body_class, height_class

# ── ANA RENDER FONKSİYONU ─────────────────────────────────
def render_avatar(height_cm, weight_kg, gender="male",
                  skin_hex=None, bg_color=None,
                  output_path=None):
    """
    Tam pipeline:
    1. Boy/kilo → avatar seç
    2. Arka plan kaldır (rembg)
    3. Ten rengi uygula (Reinhard)
    4. Arka plan oluştur
    5. Yüz blur
    6. Birleştir

    Parametreler:
    - height_cm: boy (cm)
    - weight_kg: kilo (kg)
    - gender: "male" / "female"
    - skin_hex: "#C68642" ten rengi hex (None = değiştirme)
    - bg_color: None/"white"/"black"/"#RRGGBB"
    - output_path: kayıt yolu (None = kaydetme)
    """
    # Avatar seç
    avatar_path, body_class, height_class = select_avatar(height_cm, weight_kg, gender)

    if not os.path.exists(avatar_path):
        print(f"⚠ Avatar bulunamadı: {avatar_path}")
        return None

    # Arka plan kaldır
    try:
        from rembg import remove as rembg_remove
        img = Image.open(avatar_path).convert("RGB")
        img_removed = rembg_remove(img)  # RGBA
    except Exception as e:
        print(f"rembg hatası: {e} — orijinal kullanılıyor")
        img_removed = Image.open(avatar_path).convert("RGBA")

    w, h = img_removed.size
    arr = np.array(img_removed.convert("RGB"))

    # Ten rengi
    if skin_hex:
        arr = reinhard_skin_transfer(arr, skin_hex)

    # Arka plan oluştur
    bg = create_background(w, h, color=bg_color)
    bg_img = Image.fromarray(bg).convert("RGBA")

    # Birleştir
    fg = Image.fromarray(arr).convert("RGBA")
    if img_removed.mode == "RGBA":
        fg.putalpha(img_removed.split()[3])
    bg_img.paste(fg, (0, 0), fg)
    result_arr = np.array(bg_img.convert("RGB"))

    # Yüz blur
    result_arr = apply_face_blur(result_arr)

    result = Image.fromarray(result_arr)
    if output_path:
        result.save(output_path, quality=95)
        print(f"✓ Kaydedildi: {output_path}")

    return result

print("✓ avatar_renderer hazır")
print("Kullanım: render_avatar(height_cm=178, weight_kg=75, gender='male', skin_hex='#C68642')")
