import torch, os, clip, numpy as np, colorsys, itertools, uuid, shutil, random
from PIL import Image
from scipy import ndimage
import torch.nn.functional as F
from sklearn.cluster import KMeans
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn, asyncio, threading, time, requests
from datetime import datetime
import nest_asyncio
nest_asyncio.apply()

# ── MODELLER ──────────────────────────────────────────────
print("Modeller yükleniyor...")
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
seg_model = SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes").to("cuda").eval()
clip_model, clip_preprocess = clip.load("ViT-L/14", device="cuda")
print("✓ Modeller hazır")

CLOTHING_LABELS = {4:'Upper-clothes',5:'Skirt',6:'Pants',7:'Dress',9:'Left-shoe',10:'Right-shoe'}
CONF_THRESHOLD  = 0.50
SEG_DIR         = "/content/data/segmented_items"
os.makedirs(SEG_DIR, exist_ok=True)
WARDROBE_STORE  = {}

# ── MODÜL 1: SEGMENTATION ─────────────────────────────────
def clean_mask(mask, ratio=0.15):
    labeled, n = ndimage.label(mask)
    if n <= 1: return mask
    sizes = ndimage.sum(mask, labeled, range(1, n+1))
    mx = max(sizes); out = np.zeros_like(mask)
    for i, s in enumerate(sizes):
        if s >= mx * ratio: out[labeled==(i+1)] = 1
    return out.astype(bool)

def calc_conf(mask, h, w):
    ratio = mask.sum()/(h*w)
    _, pieces = ndimage.label(mask)
    rows, cols = np.any(mask,axis=1), np.any(mask,axis=0)
    if not rows.any(): return 0
    rmin,rmax = np.where(rows)[0][[0,-1]]
    cmin,cmax = np.where(cols)[0][[0,-1]]
    bbox = (rmax-rmin)*(cmax-cmin)
    fill = mask.sum()/bbox if bbox>0 else 0
    return round(min(ratio*20,1)*0.3+(1.0 if pieces==1 else 0.4)*0.3+fill*0.4, 2)

def segment_clothing(image_path, is_real=False):
    img = Image.open(image_path).convert("RGB")
    w,h = img.size
    if max(w,h)>1024:
        r=1024/max(w,h); img=img.resize((int(w*r),int(h*r)),Image.LANCZOS)
    orig_w,orig_h = img.size
    inputs = processor(images=img,return_tensors="pt").to("cuda")
    with torch.no_grad(): outputs = seg_model(**inputs)
    seg = F.interpolate(outputs.logits,size=(orig_h,orig_w),mode='bilinear',align_corners=False)
    seg_map = seg.argmax(dim=1).squeeze().cpu().numpy()
    img_np = np.array(img)
    results, rejected = {}, {}
    for lid, lname in CLOTHING_LABELS.items():
        mask = (seg_map==lid)
        if mask.sum() < (300 if is_real else 500): continue
        if lid in [9,10]:
            s=int(orig_h*0.80); r=np.zeros_like(mask); r[s:,:]=mask[s:,:]; mask=r
            if mask.sum()<100: continue
        mask = clean_mask(mask)
        if mask.sum()<100: continue
        mask = ndimage.binary_closing(mask,iterations=4 if is_real else 3)
        conf = calc_conf(mask,orig_h,orig_w)
        thr  = CONF_THRESHOLD-0.05 if is_real else CONF_THRESHOLD
        if conf<thr: rejected[lname]=conf; continue
        rgba=np.zeros((orig_h,orig_w,4),dtype=np.uint8)
        rgba[:,:,:3]=img_np; rgba[:,:,3]=(mask*255).astype(np.uint8)
        rows,cols=np.any(mask,axis=1),np.any(mask,axis=0)
        rmin,rmax=np.where(rows)[0][[0,-1]]; cmin,cmax=np.where(cols)[0][[0,-1]]
        p=20 if is_real else 15
        crop=rgba[max(0,rmin-p):min(orig_h,rmax+p),max(0,cmin-p):min(orig_w,cmax+p)]
        results[lname]={"image":Image.fromarray(crop.astype(np.uint8)),"confidence":conf}
    return results, rejected, seg_map

# ── MODÜL 2: FEATURE EXTRACTION ───────────────────────────
def clip_embed(pil_rgba):
    bg=Image.new("RGB",pil_rgba.size,(255,255,255))
    bg.paste(pil_rgba,mask=pil_rgba.split()[3])
    t=clip_preprocess(bg).unsqueeze(0).to("cuda")
    with torch.no_grad():
        e=clip_model.encode_image(t); e=e/e.norm(dim=-1,keepdim=True)
    return e.cpu().numpy().flatten().tolist()

def color_palette(pil_rgba, k=5):
    arr=np.array(pil_rgba); alpha=arr[:,:,3]
    px=arr[:,:,:3][alpha>128]
    if len(px)<k: return []
    km=KMeans(n_clusters=k,random_state=42,n_init=10); km.fit(px)
    out=[]
    for i,c in enumerate(km.cluster_centers_):
        w=(km.labels_==i).sum()/len(km.labels_)
        h,s,v=colorsys.rgb_to_hsv(*(c/255.))
        out.append({"h":round(h*360,1),"s":round(s,3),"v":round(v,3),"weight":round(w,3)})
    return sorted(out,key=lambda x:x["weight"],reverse=True)

def get_category(label):
    return {"Upper-clothes":"top","Skirt":"bottom","Pants":"bottom",
            "Dress":"full_body","Left-shoe":"footwear","Right-shoe":"footwear"}.get(label,"unknown")

def detect_outerwear(pil_rgba, label):
    if label!="Upper-clothes": return False,0
    bg=Image.new("RGB",pil_rgba.size,(255,255,255)); bg.paste(pil_rgba,mask=pil_rgba.split()[3])
    t=clip_preprocess(bg).unsqueeze(0).to("cuda")
    prompts=["a jacket or coat or outerwear or puffer or blazer",
             "a t-shirt or hoodie or sweatshirt or shirt or knitwear"]
    tok=clip.tokenize(prompts).to("cuda")
    with torch.no_grad():
        i=clip_model.encode_image(t); tx=clip_model.encode_text(tok)
        i=i/i.norm(dim=-1,keepdim=True); tx=tx/tx.norm(dim=-1,keepdim=True)
        sim=(i@tx.T).squeeze()
    return sim[0].item()>sim[1].item(), round(sim[0].item()-sim[1].item(),3)

def detect_subcategory(pil_rgba, label):
    bg=Image.new("RGB",pil_rgba.size,(255,255,255)); bg.paste(pil_rgba,mask=pil_rgba.split()[3])
    t=clip_preprocess(bg).unsqueeze(0).to("cuda")
    if label=="Upper-clothes":
        p=["casual t-shirt or basic tee","dress shirt or button-up, formal","hoodie or sweatshirt","knit sweater or knitwear"]
        types=["tshirt","shirt","hoodie","knitwear"]; fabrics=["cotton","formal_fabric","fleece","wool"]; form=[0.2,0.7,0.3,0.5]
    elif label=="Pants":
        p=["formal trousers or dress pants","jeans or denim","sweatpants or joggers","shorts","chinos or casual trousers"]
        types=["formal_pants","jeans","sweatpants","shorts","chinos"]; fabrics=["wool","denim","fleece","light","cotton"]; form=[0.8,0.4,0.1,0.2,0.5]
    elif label=="Skirt":
        p=["mini skirt","midi skirt","maxi skirt flowing"]
        types=["mini_skirt","midi_skirt","maxi_skirt"]; fabrics=["light","medium","flowing"]; form=[0.3,0.5,0.6]
    elif label in ["Left-shoe","Right-shoe"]:
        p=["formal dress shoes loafers oxford leather","sneakers sports shoes rubber","boots ankle boots","sandals slip-ons"]
        types=["formal","sneakers","boots","sandals"]; fabrics=["leather","rubber","leather","light"]; form=[0.9,0.2,0.5,0.1]
    elif label=="Dress":
        p=["casual dress cotton","formal cocktail dress elegant","maxi dress flowing"]
        types=["casual_dress","formal_dress","maxi_dress"]; fabrics=["cotton","elegant","flowing"]; form=[0.3,0.8,0.5]
    else:
        return {"type":"unknown","fabric":"unknown","formality":0.5,"conf":0}
    tok=clip.tokenize(p).to("cuda")
    with torch.no_grad():
        i=clip_model.encode_image(t); tx=clip_model.encode_text(tok)
        i=i/i.norm(dim=-1,keepdim=True); tx=tx/tx.norm(dim=-1,keepdim=True)
        sim=(i@tx.T).squeeze()
    idx=sim.argmax().item()
    return {"type":types[idx],"fabric":fabrics[idx],"formality":form[idx],"conf":round(sim[idx].item(),3)}

def extract_features(item_id, label, pil_rgba, conf):
    return {"item_id":item_id,"label":label,"category":get_category(label),
            "confidence":conf,"clip_embedding":clip_embed(pil_rgba),"color_palette":color_palette(pil_rgba)}

# ── MODÜL 3: OUTFIT ENGINE ────────────────────────────────
OCC_KW = {
    "formal":      ["toplantı","iş","ofis","sunum","müşteri","profesyonel","resmi","görüşme","mülakat","konferans"],
    "sport":       ["spor","koşu","gym","antrenman","basketbol","futbol","egzersiz","voleybol","tenis","pilates","yoga","fitness","basket","maç","sahaya","hike","trekking"],
    "outdoor":     ["doğa","kamp","yürüyüş","piknik","orman","dağ","deniz","plaj","açık hava","park","gezi","tur"],
    "school":      ["okul","ders","üniversite","kampüs","sınav","kütüphane","dershane","kurs","staj"],
    "event_casual":["konser","festival","sergi","galeri","açılış","müze","tiyatro"],
    "event":       ["düğün","nişan","davet","gala","parti","özel","mezuniyet","kutlama"],
    "night":       ["gece","bar","kulüp","akşam yemeği","randevu","date","romantik","lounge"],
    "casual":      ["arkadaş","buluşma","günlük","rahat","kahve","alışveriş","market","sade","gezinti","brunch","hafta sonu"],
}
OCC_FALLBACK  = {"outdoor":"casual","school":"casual","event_casual":"casual"}
OCC_FORMALITY = {"formal":0.6,"event":0.65,"night":0.4,"event_casual":0.2,"outdoor":0.0,"school":0.0,"casual":0.0,"sport":0.0}
OCC_BOTTOM    = {"sport":["sweatpants","shorts"],"outdoor":["jeans","chinos","sweatpants"],"formal":["formal_pants","chinos"],
                 "event":["formal_pants","chinos","midi_skirt","maxi_skirt"],"event_casual":["jeans","chinos","mini_skirt"],
                 "night":["formal_pants","jeans","chinos","mini_skirt"],"school":["jeans","chinos","sweatpants"],"casual":None}
OCC_OUTER     = {"formal":["shirt"],"event":["shirt","knitwear"],"event_casual":None,"outdoor":["hoodie","knitwear"],
                 "sport":[],"casual":None,"school":None,"night":["shirt","knitwear"]}
OCC_SHOE      = {"sport":["sneakers"],"outdoor":["sneakers","boots"],"school":["sneakers","boots","formal"],
                 "casual":["sneakers","boots","formal","sandals"],"event_casual":["sneakers","boots","formal"],
                 "night":["formal","boots"],"formal":["formal"],"event":["formal","boots"]}
FB_OK         = {"casual_dress":["casual","school","event_casual","outdoor"],"formal_dress":["event","night","formal"],"maxi_dress":["event","casual","night","event_casual"]}
SK_OK         = {"mini_skirt":["casual","night","event_casual"],"midi_skirt":["casual","event_casual","night","event","formal"],"maxi_skirt":["casual","event","event_casual","night"]}
STYLE_KW      = {"minimal":["sade","minimal","basit","temiz"],"smart_casual":["uğraşılmış","özenli","şık"],"sporty":["spor","atletik","aktif"],"streetwear":["sokak","street","oversize"]}
WEATHER_KW    = {"cold":["üşüdüm","soğuk","donuyorum","kışlık"],"warm":["sıcak","bunaltıcı","yazlık"],"rainy":["yağmur","ıslak","yağışlı"],"mild":["ılık","normal"]}
TIME_KW       = {"morning":["sabah","kahvaltı","erken"],"afternoon":["öğle","öğleden sonra","gündüz"],"evening":["akşam","akşamüstü"],"night":["gece","geç"]}

def time_ctx(hour=None):
    h=hour or datetime.now().hour
    if 6<=h<12: return "morning"
    elif 12<=h<17: return "afternoon"
    elif 17<=h<21: return "evening"
    else: return "night"

def temp_layer(t):
    if t>=25: return "light"
    elif t>=15: return "mid"
    elif t>=5: return "heavy"
    return "extreme"

def parse_prompt(prompt, hour=None):
    p=prompt.lower(); occ=None
    for o,kws in OCC_KW.items():
        if any(k in p for k in kws): occ=o; break
    unknown=occ is None
    if unknown: occ="casual"
    disp=occ; occ=OCC_FALLBACK.get(occ,occ)
    sty=next((s for s,kws in STYLE_KW.items() if any(k in p for k in kws)),None)
    wth=next((w for w,kws in WEATHER_KW.items() if any(k in p for k in kws)),None)
    tc=time_ctx(hour)
    for t,kws in TIME_KW.items():
        if any(k in p for k in kws): tc=t; break
    return {"occasion":occ,"display_occasion":disp,"style_hint":sty,"weather_hint":wth,"time_context":tc,"is_unknown":unknown,"raw_prompt":prompt}

def color_ok(p1,p2):
    if not p1 or not p2: return True,"no_palette"
    d1,d2=p1[0],p2[0]
    if d1["s"]<0.15 or d2["s"]<0.15: return True,"neutral"
    dist=min(abs(d1["h"]-d2["h"]),360-abs(d1["h"]-d2["h"]))
    if 30<dist<150: return False,f"clash"
    return True,"ok"

def outfit_complete(cats): return "full_body" in cats or ("top" in cats and "bottom" in cats)

def clip_score(combo):
    embs=[np.array(f["clip_embedding"]) for f in combo]
    if len(embs)<2: return 0.0
    scores=[np.dot(e1,e2)/(np.linalg.norm(e1)*np.linalg.norm(e2)) for e1,e2 in itertools.combinations(embs,2)]
    return round(float(np.mean(scores)),4)

def formality_score(combo):
    fs=[f.get("formality",0.5) for f in combo]
    if len(fs)<2: return 1.0
    return round(1.0-np.mean([abs(a-b) for a,b in itertools.combinations(fs,2)]),3)

def ctx_filter(combo, weather, occasion, tc):
    cats=[f["category"] for f in combo]
    if weather:
        layer=temp_layer(weather.get("temp_c",20))
        for f in combo:
            if layer=="light" and f["category"]=="outerwear": return False
            if layer in ["heavy","extreme"] and f["category"]=="top" and "outerwear" not in cats: return False
    if occasion=="sport" and any(f["label"]=="Dress" for f in combo): return False
    return True

def style_axis(o):
    avg=np.mean(o["formality"])
    if avg>=0.6: return "formal"
    elif avg>=0.4: return "smart_casual"
    elif avg>=0.25: return "casual"
    return "sporty"

def bottom_ok(f, disp, occ, layer, tc, pref, min_f):
    sub=f.get("subcategory","?")
    if "skirt" in sub:
        ok=SK_OK.get(sub,[])
        if disp not in ok and occ not in ok: return False
    if sub=="shorts" and (layer in ["heavy","extreme"] or tc=="morning"): return False
    if sub=="sweatpants" and disp in ["formal","event","night"]: return False
    if pref: return sub in pref
    return f.get("formality",0.5)>=min_f if min_f>0 else True

def generate(all_features, prompt, weather=None, hour=None, top_k=3):
    ctx=parse_prompt(prompt,hour)
    if weather is None:
        t={"cold":8,"mild":18,"warm":28,"rainy":14,None:20}.get(ctx["weather_hint"],20)
        weather={"temp_c":t,"rain":ctx["weather_hint"]=="rainy"}
    occ=ctx["occasion"]; disp=ctx["display_occasion"]; tc=ctx["time_context"]
    min_f=OCC_FORMALITY.get(disp,0.0); layer=temp_layer(weather.get("temp_c",20))

    print(f"  → {disp} | {ctx['style_hint']} | {ctx['weather_hint']} | {tc}"
          +(" [bilinmeyen→casual]" if ctx["is_unknown"] else ""))

    by={}
    for f in all_features: by.setdefault(f["category"],[]).append(f)

    def fok(f): return f.get("formality",0.5)>=min_f

    # Footwear
    ash=OCC_SHOE.get(disp)
    fw=[f for f in by.get("footwear",[]) if
        (not ash or f.get("subcategory") in ash) and
        not (f.get("subcategory")=="sandals" and (layer in ["heavy","extreme"] or tc=="morning")) and
        (fok(f) if min_f>0 else True)] or by.get("footwear",[])
    seen_s,ufw=set(),[]
    for f in fw:
        s=f.get("subcategory","?")
        if s not in seen_s: seen_s.add(s); ufw.append(f)
    fw=ufw

    # Bottom
    bp=OCC_BOTTOM.get(disp)
    bots=[f for f in by.get("bottom",[]) if bottom_ok(f,disp,occ,layer,tc,bp,min_f)] or by.get("bottom",[])

    # Outerwear
    op=OCC_OUTER.get(disp)
    ao=by.get("outerwear",[])
    if op is not None:
        outer=[] if not op else ([f for f in ao if f.get("subcategory") in op] or ao)
    else:
        outer=[f for f in ao if fok(f)] if min_f>0 else ao

    tops=[f for f in by.get("top",[]) if fok(f)] if min_f>0 else by.get("top",[])

    def fbok(f):
        s=f.get("subcategory","?"); ok=FB_OK.get(s,["casual"])
        return disp in ok or occ in ok
    fb=[f for f in by.get("full_body",[]) if fbok(f)]

    cands,seen=[],set()
    def add(combo):
        k=frozenset(f["item_id"] for f in combo)
        if k not in seen: seen.add(k); cands.append(combo)

    for shoe in (fw or [None]):
        sl=[shoe] if shoe else []
        for t,b in itertools.product(tops,bots): add([t,b]+sl)
        for o,t,b in itertools.product(outer,tops,bots):
            if o.get("subcategory")==t.get("subcategory"): continue
            add([o,t,b]+sl)
        for f in fb: add([f]+sl)

    valid=[]
    for combo in cands:
        if not outfit_complete([f["category"] for f in combo]): continue
        if not ctx_filter(combo,weather,occ,tc): continue
        ns=[f for f in combo if f["category"]!="footwear"]
        if not all(color_ok(f1["color_palette"],f2["color_palette"])[0]
                   for f1,f2 in itertools.combinations(ns,2)): continue
        cs=clip_score(combo); fs=formality_score(combo)
        valid.append({
            "items":[f["item_id"] for f in combo],
            "labels":[f.get("subcategory",f["label"]) for f in combo],
            "categories":[f["category"] for f in combo],
            "fabrics":[f.get("fabric","?") for f in combo],
            "formality":[f.get("formality",0.5) for f in combo],
            "final_score":round(cs*0.65+fs*0.35,4),
            "context":ctx,
        })

    if not valid:
        msgs={"formal":"Formal kıyafet yetersiz. Gömlek veya kumaş pantolon ekle.",
              "event":"Özel etkinlik için yeterli parça yok.","night":"Gece için yeterli parça bulunamadı."}
        return [{"items":[],"labels":[],"categories":[],"fabrics":[],"formality":[],
                 "final_score":0,"style_axis":"","message":msgs.get(disp,"Uyumlu kombin bulunamadı."),"context":ctx}]

    valid.sort(key=lambda x:x["final_score"],reverse=True)
    pool=valid[:25]; selected=[]
    for c in pool:
        if len(selected)>=top_k: break
        ci=set(c["items"]); cl=set(c["labels"])
        if not any(len(ci&set(s["items"]))/len(ci|set(s["items"]))>0.5 or
                   len(cl&set(s["labels"]))/len(cl|set(s["labels"]))>0.75 for s in selected):
            c["style_axis"]=style_axis(c); selected.append(c)
    for o in pool:
        if len(selected)>=top_k: break
        if o not in selected: o["style_axis"]=style_axis(o); selected.append(o)
    return selected

# ── MODÜL 4: FASTAPI ──────────────────────────────────────
app=FastAPI(title="Aura",version="1.1")

class OutfitReq(BaseModel):
    user_id:str; weather:Optional[dict]={"temp_c":20,"rain":False}
    occasion:Optional[str]="casual"; hour:Optional[int]=None; prompt:Optional[str]=None

class FeedbackReq(BaseModel):
    user_id:str; outfit_items:list; action:str

@app.get("/health")
def health(): return {"status":"ok","version":"1.1"}

@app.post("/wardrobe/upload")
async def upload(user_id:str, file:UploadFile=File(...), real_photo:bool=False):
    iid=str(uuid.uuid4())
    fn=file.filename.lower()
    suf=".avif" if fn.endswith(".avif") else ".png" if fn.endswith(".png") else ".jpg"
    tmp=f"/tmp/{iid}{suf}"
    with open(tmp,"wb") as f: shutil.copyfileobj(file.file,f)
    try:
        img=Image.open(tmp).convert("RGB")
        if suf==".avif": tmp=f"/tmp/{iid}.jpg"; img.save(tmp,"JPEG")
    except Exception as e: raise HTTPException(422,f"Görsel açılamadı: {e}")
    try: results,rejected,_=segment_clothing(tmp,is_real=real_photo)
    except Exception as e: raise HTTPException(422,f"Segmentation hatası: {e}")
    if not results:
        return JSONResponse(422,{"status":"failed","message":"Kıyafet bulunamadı. Farklı açıdan tekrar çek."})
    items=WARDROBE_STORE.setdefault(user_id,[]); proc,retry=[],[]
    for lname,data in results.items():
        fid=f"{user_id}_{iid}_{lname.lower().replace('-','_').replace(' ','_')}"
        seg_path=os.path.join(SEG_DIR,f"{fid}.png"); data["image"].save(seg_path)
        is_o,_=detect_outerwear(data["image"],lname)
        cat="outerwear" if (lname=="Upper-clothes" and is_o) else get_category(lname)
        sub=detect_subcategory(data["image"],lname)
        feat=extract_features(fid,lname,data["image"],data["confidence"])
        feat.update({"category":cat,"subcategory":sub["type"],"fabric":sub["fabric"],"formality":sub["formality"],"seg_path":seg_path})
        items.append(feat)
        proc.append({"item_id":fid,"category":cat,"subcategory":sub["type"],"fabric":sub["fabric"],"formality":sub["formality"],"confidence":data["confidence"]})
    for lname,conf in rejected.items():
        retry.append({"label":lname,"message":f"{lname} net görünmüyor ({conf:.2f}). Ayrı çek."})
    return {"status":"ok","user_id":user_id,"processed":proc,"retry_needed":retry}

@app.get("/wardrobe/{user_id}")
def get_wardrobe(user_id:str):
    items=WARDROBE_STORE.get(user_id,[])
    return {"user_id":user_id,"count":len(items),
            "items":[{"item_id":f["item_id"],"category":f["category"],"subcategory":f.get("subcategory","?"),
                      "fabric":f.get("fabric","?"),"confidence":f["confidence"]} for f in items]}

@app.post("/outfit/generate")
def gen_outfit(req:OutfitReq):
    items=WARDROBE_STORE.get(req.user_id,[])
    if not items: raise HTTPException(404,"Dolap boş.")
    if len(items)<2: raise HTTPException(422,"En az 2 kıyafet gerekli.")
    outfits=generate(items,req.prompt or req.occasion or "casual",req.weather,req.hour)
    if not outfits or not outfits[0].get("items"):
        return {"status":"no_match","message":outfits[0].get("message","Kombin bulunamadı."),"outfits":[]}
    return {"status":"ok","user_id":req.user_id,"outfits":outfits}

@app.post("/outfit/feedback")
def feedback(req:FeedbackReq):
    if req.action not in ["like","dislike","worn"]: raise HTTPException(422,"action: like|dislike|worn")
    return {"status":"ok"}

# ── SERVER ────────────────────────────────────────────────
def run():
    config=uvicorn.Config(app,host="0.0.0.0",port=8000,log_level="warning")
    uvicorn.Server(config).run() # blocking — run in thread

if __name__=="__main__":
    t=threading.Thread(target=run,daemon=True); t.start(); time.sleep(3)
    print(f"✓ API: {requests.get('http://localhost:8000/health').json()}")

# ── YENİ: RENK + SKOR SİSTEMİ ────────────────────────────

def color_score_pair(p1, p2):
    if not p1 or not p2: return 0.75
    d1, d2 = p1[0], p2[0]
    if d1["s"] < 0.12 and d2["s"] < 0.12: return 1.0
    if d1["s"] < 0.12 or d2["s"] < 0.12: return 0.95
    dist = min(abs(d1["h"]-d2["h"]), 360-abs(d1["h"]-d2["h"]))
    if dist <= 30:     return 1.0
    if dist <= 60:     return 0.88
    if 150<=dist<=210: return 0.80
    if 60<dist<=120:   return 0.40
    return 0.25

def outfit_color_score(combo, sunny_boost=0.0):
    ns = [f for f in combo if f["category"] != "footwear"]
    if len(ns) < 2: return 0.8
    scores = [color_score_pair(f1["color_palette"], f2["color_palette"])
              for f1,f2 in itertools.combinations(ns,2)]
    return min(1.0, round(np.mean(scores),3) + sunny_boost)

def color_hard_ok(combo):
    ns = [f for f in combo if f["category"] != "footwear"]
    for f1,f2 in itertools.combinations(ns,2):
        if color_score_pair(f1["color_palette"],f2["color_palette"]) < 0.35:
            return False
    return True

def get_sunny_boost(weather):
    if not weather: return 0.0
    if weather.get("rain"): return 0.0
    if weather.get("sunny") or weather.get("temp_c",20) >= 22: return 0.06
    return 0.0

def formality_gap_ok(combo):
    fs = [f.get("formality",0.5) for f in combo if f["category"] != "footwear"]
    if len(fs) < 2: return True
    return max(fs) - min(fs) <= 0.45

def style_mix_ok(combo):
    tops = [f for f in combo if f["category"] in ["top","outerwear"]]
    bottoms = [f for f in combo if f["category"] == "bottom"]
    for t in tops:
        for b in bottoms:
            if t.get("subcategory") in ["hoodie","tshirt"] and "skirt" in b.get("subcategory",""):
                return False
    return True

def shoe_bottom_ok(combo):
    shoes = [f.get("subcategory") for f in combo if f["category"]=="footwear"]
    bottoms = [f.get("subcategory") for f in combo if f["category"]=="bottom"]
    if "sandals" in shoes and "shorts" in bottoms: return False
    return True

def outerwear_occasion_ok(f, disp, min_f):
    if f["category"] != "outerwear": return True
    if disp in ["casual","school","outdoor","event_casual"]:
        return f.get("formality",0.5) <= 0.65
    return True

def texture_score(combo):
    fabrics = [f.get("fabric","?") for f in combo if f["category"] != "footwear"]
    heavy = ["wool","fleece","leather"]
    heavy_count = sum(1 for f in fabrics if f in heavy)
    if not fabrics: return 0.5
    return round(1.0 - abs(heavy_count/len(fabrics) - 0.5), 3)

def final_score(combo, sunny_boost=0.0):
    cs  = clip_score(combo)
    col = outfit_color_score(combo, sunny_boost)
    tex = texture_score(combo)
    return round(0.6*cs + 0.3*col + 0.1*tex, 4)

def weighted_random_select(pool, top_k=3):
    if len(pool) <= top_k: return pool
    tier1 = pool[:3]
    tier2 = pool[3:6] if len(pool)>3 else []
    tier3 = pool[6:10] if len(pool)>6 else []
    selected = []
    attempts = 0
    while len(selected) < top_k and attempts < 50:
        attempts += 1
        r = random.random()
        if r < 0.60 and tier1: candidate = random.choice(tier1)
        elif r < 0.90 and tier2: candidate = random.choice(tier2)
        elif tier3: candidate = random.choice(tier3)
        else: candidate = random.choice(pool)
        if candidate not in selected:
            selected.append(candidate)
    return selected

def generate(all_features, prompt, weather=None, hour=None, top_k=3):
    ctx = parse_prompt(prompt, hour)
    if weather is None:
        t = {"cold":8,"mild":18,"warm":28,"rainy":14,None:20}.get(ctx["weather_hint"],20)
        weather = {"temp_c":t,"rain":ctx["weather_hint"]=="rainy"}
    occ=ctx["occasion"]; disp=ctx["display_occasion"]
    tc=ctx["time_context"]
    min_f=OCC_FORMALITY.get(disp,0.0)
    layer=temp_layer(weather.get("temp_c",20))
    sunny_boost=get_sunny_boost(weather)

    print(f"  → {disp} | {ctx['style_hint']} | {ctx['weather_hint']} | {tc}"
          +(" ☀+renk" if sunny_boost>0 else "")
          +(" [bilinmeyen→casual]" if ctx["is_unknown"] else ""))

    by={}
    for f in all_features: by.setdefault(f["category"],[]).append(f)
    def fok(f): return f.get("formality",0.5)>=min_f

    ash=OCC_SHOE.get(disp)
    fw=[f for f in by.get("footwear",[]) if
        (not ash or f.get("subcategory") in ash) and
        not (f.get("subcategory")=="sandals" and (layer in ["heavy","extreme"] or tc=="morning")) and
        (fok(f) if min_f>0 else True)] or by.get("footwear",[])
    seen_s,ufw=set(),[]
    for f in fw:
        s=f.get("subcategory","?")
        if s not in seen_s: seen_s.add(s); ufw.append(f)
    fw=ufw

    bp=OCC_BOTTOM.get(disp)
    bots=[f for f in by.get("bottom",[]) if bottom_ok(f,disp,occ,layer,tc,bp,min_f)] or by.get("bottom",[])
    op=OCC_OUTER.get(disp); ao=by.get("outerwear",[])
    if op is not None:
        outer=[] if not op else ([f for f in ao if f.get("subcategory") in op] or ao)
    else:
        outer=[f for f in ao if fok(f) and outerwear_occasion_ok(f,disp,min_f)] if min_f>0 else               [f for f in ao if outerwear_occasion_ok(f,disp,min_f)]

    tops=[f for f in by.get("top",[]) if fok(f)] if min_f>0 else by.get("top",[])
    def fbok(f):
        s=f.get("subcategory","?"); ok=FB_OK.get(s,["casual"])
        return disp in ok or occ in ok
    fb=[f for f in by.get("full_body",[]) if fbok(f)]

    cands,seen=[],set()
    def add(combo):
        k=frozenset(f["item_id"] for f in combo)
        if k not in seen: seen.add(k); cands.append(combo)

    for shoe in (fw or [None]):
        sl=[shoe] if shoe else []
        for t,b in itertools.product(tops,bots): add([t,b]+sl)
        for o,t,b in itertools.product(outer,tops,bots):
            if o.get("subcategory")==t.get("subcategory"): continue
            add([o,t,b]+sl)
        for f in fb: add([f]+sl)

    valid=[]
    for combo in cands:
        if not outfit_complete([f["category"] for f in combo]): continue
        if not ctx_filter(combo,weather,occ,tc): continue
        if not formality_gap_ok(combo): continue
        if not style_mix_ok(combo): continue
        if not shoe_bottom_ok(combo): continue
        if not color_hard_ok(combo): continue
        cs=clip_score(combo)
        if cs<0.18: continue
        sc=final_score(combo,sunny_boost)
        if sc<0.62: continue
        valid.append({
            "items":[f["item_id"] for f in combo],
            "labels":[f.get("subcategory",f["label"]) for f in combo],
            "categories":[f["category"] for f in combo],
            "fabrics":[f.get("fabric","?") for f in combo],
            "formality":[f.get("formality",0.5) for f in combo],
            "final_score":sc,"context":ctx,
        })

    if not valid:
        msgs={"formal":"Formal kıyafet yetersiz. Gömlek veya kumaş pantolon ekle.",
              "event":"Özel etkinlik için yeterli parça yok.",
              "night":"Gece için yeterli parça bulunamadı."}
        return [{"items":[],"labels":[],"categories":[],"fabrics":[],"formality":[],
                 "final_score":0,"style_axis":"","message":msgs.get(disp,"Uyumlu kombin bulunamadı."),"context":ctx}]

    valid.sort(key=lambda x:x["final_score"],reverse=True)
    pool=valid[:10]
    top_score=pool[0]["final_score"]
    show_k=top_k if top_score>=0.80 else (min(2,top_k) if top_score>=0.70 else 1)

    if disp in ["sport","formal","event"]:
        selected=pool[:show_k]
    else:
        selected=weighted_random_select(pool,top_k=show_k)

    final_selected=[]
    for c in selected:
        ci=set(c["items"]); cl=set(c["labels"])
        if not any(len(ci&set(s["items"]))/len(ci|set(s["items"]))>0.5 or
                   len(cl&set(s["labels"]))/len(cl|set(s["labels"]))>0.75
                   for s in final_selected):
            c["style_axis"]=style_axis(c)
            final_selected.append(c)
    return final_selected

# ── SUPABASE ──────────────────────────────────────────────
from supabase import create_client

SUPABASE_URL = "https://xzobaigtiueobqkielnn.supabase.co"
SUPABASE_KEY = "eyJhbGc..."  # anon key

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def create_user(gender="male", skin_tone="medium", avatar_key="avatar_m_medium"):
    result = supabase.table("users").insert({
        "gender": gender, "skin_tone": skin_tone, "avatar_key": avatar_key
    }).execute()
    return result.data[0]["id"]

def save_wardrobe_item(user_id, feature_dict, seg_image_path=None):
    import os
    seg_url = None
    if seg_image_path and os.path.exists(seg_image_path):
        filename = f"{user_id}/{os.path.basename(seg_image_path)}"
        with open(seg_image_path, "rb") as f:
            supabase.storage.from_("wardrobe").upload(filename, f)
        seg_url = supabase.storage.from_("wardrobe").get_public_url(filename)
    item = {
        "user_id": user_id,
        "category": feature_dict.get("category"),
        "subcategory": feature_dict.get("subcategory"),
        "fabric": feature_dict.get("fabric"),
        "formality": feature_dict.get("formality"),
        "confidence": feature_dict.get("confidence"),
        "color_palette": feature_dict.get("color_palette"),
        "seg_image_url": seg_url,
    }
    result = supabase.table("wardrobe_items").insert(item).execute()
    return result.data[0]["id"]

def load_wardrobe(user_id):
    result = supabase.table("wardrobe_items").select("*").eq("user_id", user_id).execute()
    return result.data

def init_user_session(user_id):
    items = load_wardrobe(user_id)
    wardrobe = []
    for item in items:
        feat = {
            "item_id": item["id"],
            "category": item["category"],
            "subcategory": item["subcategory"],
            "fabric": item["fabric"],
            "formality": item["formality"] or 0.5,
            "confidence": item["confidence"] or 0.8,
            "color_palette": item["color_palette"] or [],
            "clip_embedding": [],
            "seg_path": None,
            "label": item["subcategory"]
        }
        wardrobe.append(feat)
    WARDROBE_STORE[user_id] = wardrobe
    print(f"✓ Session başlatıldı — {len(wardrobe)} kıyafet yüklendi")
    return wardrobe

EXPECTED_ASPECT = {
    "Upper-clothes": (0.7, 1.6), "Pants": (1.4, 3.5),
    "Skirt": (0.8, 2.0), "Dress": (1.5, 3.5),
    "Left-shoe": (0.4, 1.2), "Right-shoe": (0.4, 1.2),
}
MIN_AREA_RATIO = {
    "Upper-clothes": 0.04, "Pants": 0.04, "Skirt": 0.03,
    "Dress": 0.05, "Left-shoe": 0.005, "Right-shoe": 0.005,
}
EXCLUDE_LABELS = {1, 2, 3, 11, 14, 15, 16, 17}

def grabcut_refine(img_np, mask, bbox, pad=30):
    import cv2
    h, w = img_np.shape[:2]
    rmin, rmax, cmin, cmax = bbox
    gc = np.full(mask.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    gc[mask] = cv2.GC_PR_FGD
    gc[ndimage.binary_erosion(mask, iterations=8)] = cv2.GC_FGD
    bg = np.ones(mask.shape, dtype=bool)
    bg[max(0,rmin-pad):min(h,rmax+pad), max(0,cmin-pad):min(w,cmax+pad)] = False
    gc[bg] = cv2.GC_BGD
    try:
        cv2.grabCut(img_np, gc, None, np.zeros((1,65),np.float64),
                    np.zeros((1,65),np.float64), 5, cv2.GC_INIT_WITH_MASK)
        return ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD))
    except:
        return mask

def segment_clothing_hybrid(image_path, is_real=True):
    import cv2
    from PIL import ImageOps
    img = Image.open(image_path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    if max(w, h) > 1024:
        r = 1024 / max(w, h)
        img = img.resize((int(w*r), int(h*r)), Image.LANCZOS)
    img_np = np.array(img)
    H, W = img_np.shape[:2]
    img_area = H * W

    inputs = seg_processor(images=img, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = seg_model(**inputs)
    seg = F.interpolate(outputs.logits, size=(H, W), mode="bilinear", align_corners=False)
    seg_map = seg.argmax(dim=1).squeeze().cpu().numpy()

    exclusion = np.isin(seg_map, list(EXCLUDE_LABELS))
    sam_predictor.set_image(img_np)
    results, rejected = {}, {}

    for lid, lname in CLOTHING_LABELS.items():
        rough = (seg_map == lid)
        if rough.sum() / img_area < MIN_AREA_RATIO.get(lname, 0.01):
            if rough.sum() > 100: rejected[lname] = "too_small"
            continue
        if lid in [9, 10]:
            s = int(H * 0.75)
            nm = np.zeros_like(rough); nm[s:,:] = rough[s:,:]
            rough = nm
            if rough.sum() < 100: continue
        rows = np.any(rough, axis=1); cols = np.any(rough, axis=0)
        if not rows.any(): continue
        rmin, rmax = np.where(rows)[0][[0,-1]]
        cmin, cmax = np.where(cols)[0][[0,-1]]
        bh, bw = rmax-rmin, cmax-cmin
        if bw > 0 and lname in EXPECTED_ASPECT:
            aspect = bh / bw
            mn, mx = EXPECTED_ASPECT[lname]
            if aspect < mn*0.6 or aspect > mx*1.5:
                rejected[lname] = "wrong_shape"; continue
        et = sum([rmin<=5, rmax>=H-5, cmin<=5, cmax>=W-5])
        if (et >= 2 and lname not in ["Pants","Dress"]) or et >= 3:
            rejected[lname] = "edge_cut"; continue
        try:
            masks, scores, _ = sam_predictor.predict(
                box=np.array([cmin, rmin, cmax, rmax]),
                multimask_output=True)
            best = int(np.argmax(scores))
            mask = masks[best].astype(bool)
            sam_score = float(scores[best])
        except:
            mask = rough; sam_score = 0.0
        method = "sam" if sam_score >= 0.6 else "segformer"
        if method == "segformer": mask = rough
        mask = mask & ~exclusion
        mask = ndimage.binary_fill_holes(mask)
        if method == "sam" and mask.sum() > 500:
            r2 = np.any(mask, axis=1); c2 = np.any(mask, axis=0)
            if r2.any():
                rm2, rmx2 = np.where(r2)[0][[0,-1]]
                cm2, cmx2 = np.where(c2)[0][[0,-1]]
                mask = grabcut_refine(img_np, mask, (rm2, rmx2, cm2, cmx2))
                mask = ndimage.binary_fill_holes(mask)
        labeled, n = ndimage.label(mask)
        if n > 1:
            sizes = ndimage.sum(mask, labeled, range(1, n+1))
            mask = labeled == (int(np.argmax(sizes)) + 1)
        mask = ndimage.binary_closing(mask, iterations=3)
        mask = ndimage.binary_fill_holes(mask)
        if mask.sum() < 200:
            rejected[lname] = "fragmented"; continue
        rows = np.any(mask, axis=1); cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0,-1]]
        cmin, cmax = np.where(cols)[0][[0,-1]]
        bbox_area = (rmax-rmin) * (cmax-cmin)
        fill_ratio = mask.sum() / bbox_area if bbox_area > 0 else 0
        if fill_ratio < 0.30:
            rejected[lname] = "too_sparse"; continue
        confidence = round(sam_score * (0.6 + fill_ratio*0.4), 2) if method=="sam" else 0.65
        if confidence < 0.55:
            rejected[lname] = f"low_conf_{confidence}"; continue
        alpha = cv2.GaussianBlur((mask*255).astype(np.uint8), (3,3), 0.8)
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        rgba[:,:,:3] = img_np; rgba[:,:,3] = alpha
        p = 15
        crop = rgba[max(0,rmin-p):min(H,rmax+p), max(0,cmin-p):min(W,cmax+p)]
        results[lname] = {
            "image": Image.fromarray(crop),
            "confidence": confidence, "method": method,
            "fill_ratio": round(fill_ratio, 2),
        }
    return results, rejected, seg_map

# ── SESSION 2: CLIP EMBEDDING + MOOD BIAS ──
import io, uuid, json

def clip_embed(pil_image):
    """PIL → 768-dim L2-normalize embedding (list[float])"""
    img = pil_image.convert("RGB") if pil_image.mode != "RGB" else pil_image
    with torch.no_grad():
        x = clip_preprocess(img).unsqueeze(0).cuda()
        emb = clip_model.encode_image(x)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze().cpu().numpy().astype(float).tolist()


def save_wardrobe_item_v2(user_id, feature_dict, seg_rgba_pil):
    """Segmente kıyafet + CLIP embedding → Supabase"""
    item_id = str(uuid.uuid4())
    buf = io.BytesIO()
    seg_rgba_pil.save(buf, format="PNG", compress_level=0)
    buf.seek(0)
    path = f"{user_id}/{item_id}_seg.png"
    supabase.storage.from_("wardrobe").upload(
        path, buf.getvalue(),
        {"content-type": "image/png", "x-upsert": "true"})
    seg_url = supabase.storage.from_("wardrobe").get_public_url(path)
    embedding = feature_dict.get("clip_embedding") or clip_embed(seg_rgba_pil)
    item = {
        "id": item_id, "user_id": user_id,
        "category": feature_dict.get("category"),
        "subcategory": feature_dict.get("subcategory"),
        "fabric": feature_dict.get("fabric"),
        "formality": feature_dict.get("formality"),
        "confidence": feature_dict.get("confidence"),
        "color_palette": feature_dict.get("color_palette"),
        "seg_image_url": seg_url,
        "clip_embedding": embedding,
    }
    supabase.table("wardrobe_items").insert(item).execute()
    return item_id


def load_wardrobe_v2(user_id):
    res = supabase.table("wardrobe_items").select("*").eq("user_id", user_id).execute()
    wardrobe = []
    for item in res.data:
        emb = item.get("clip_embedding")
        if isinstance(emb, str):
            try: emb = json.loads(emb)
            except: emb = []
        elif emb is None: emb = []
        wardrobe.append({
            "item_id": item["id"], "category": item["category"],
            "subcategory": item["subcategory"], "fabric": item["fabric"],
            "formality": item.get("formality") or 0.5,
            "confidence": item.get("confidence") or 0.8,
            "color_palette": item.get("color_palette") or [],
            "clip_embedding": emb,
            "seg_path": item.get("seg_image_url"),
            "label": item["subcategory"],
        })
    return wardrobe


# ── MOOD + EVENT (V1 — sade, soft bias) ──
MOOD_LIST = ["rahat", "şık", "enerjik", "yorgun", "minimal", "cesur"]
EVENT_LIST = ["okul", "iş", "buluşma", "spor", "akşam", "evde"]

# Event = ANA FİLTRE: minimum formality eşiği
EVENT_FORMALITY = {
    "okul": 0.20, "iş": 0.55, "buluşma": 0.30,
    "spor": 0.10, "akşam": 0.40, "evde": 0.0,
}

# Mood = SOFT BIAS: skoru hafifçe eğer, kıyafet elemez
MOOD_BIAS = {
    "rahat":   {"formality": -0.12, "saturation": 0.0,   "clash_tol": 0.0},
    "şık":     {"formality": +0.18, "saturation": -0.05, "clash_tol": -0.05},
    "enerjik": {"formality": -0.05, "saturation": +0.12, "clash_tol": +0.03},
    "yorgun":  {"formality": -0.10, "saturation": -0.08, "clash_tol": 0.0},
    "minimal": {"formality":  0.0,  "saturation": -0.15, "clash_tol": -0.05},
    "cesur":   {"formality":  0.0,  "saturation": +0.10, "clash_tol": +0.10},
}


def apply_mood_score(score, combo, mood):
    """Mood'a göre skoru ±0.20 aralığında eğer. Hard filter YOK."""
    bias = MOOD_BIAS.get(mood, {})
    if not bias: return score
    avg_form = float(np.mean([f.get("formality", 0.5) for f in combo]))
    score += bias.get("formality", 0) * (avg_form - 0.5) * 2
    sats = [p.get("s", 0.3) for f in combo for p in (f.get("color_palette") or [])]
    if sats:
        avg_sat = float(np.mean(sats))
        score += bias.get("saturation", 0) * (avg_sat - 0.4) * 2
    return round(score, 4)


def apply_weather_score(score, combo, weather):
    """Hava davranış bias'ı — gerçek psikoloji. Soft."""
    if not weather: return score
    temp = weather.get("temp_c", 20)
    sunny = weather.get("sunny", False)
    rainy = weather.get("rain", False)
    sats = [p.get("s", 0.3) for f in combo for p in (f.get("color_palette") or [])]
    vals = [p.get("v", 0.5) for f in combo for p in (f.get("color_palette") or [])]
    avg_sat = float(np.mean(sats)) if sats else 0.3
    avg_val = float(np.mean(vals)) if vals else 0.5
    if sunny and temp >= 20:
        score += 0.08 * (avg_sat - 0.4) * 2
    elif temp < 10:
        score += 0.05 * (0.4 - avg_sat) * 2
    if rainy:
        score += 0.05 * (0.5 - avg_val) * 2
    return round(score, 4)

# ── SESSION 2: GENERATE V2 (mood + event tabanlı, soft bias) ──
EVENT_TO_OCCASION = {
    "okul": "casual", "iş": "formal", "buluşma": "casual",
    "spor": "sport", "akşam": "night", "evde": "casual",
}


def parse_mood_event(mood, event, hour=None):
    occ = EVENT_TO_OCCASION.get(event, "casual")
    disp = occ
    fallback = OCC_FALLBACK.get(occ, occ)
    return {
        "occasion": fallback, "display_occasion": disp,
        "mood": mood, "event": event,
        "weather_hint": None, "time_context": time_ctx(hour),
        "is_unknown": False, "style_hint": mood or "",
    }


def generate_v2(all_features, mood=None, event="evde", weather=None, hour=None, top_k=3):
    """Event = ana filtre, weather = güvenlik filtresi, mood = soft score bias."""
    ctx = parse_mood_event(mood, event, hour)
    if weather is None:
        weather = {"temp_c": 20, "rain": False, "sunny": False}
    occ = ctx["occasion"]; disp = ctx["display_occasion"]; tc = ctx["time_context"]
    min_f = OCC_FORMALITY.get(disp, 0.0); layer = temp_layer(weather.get("temp_c", 20))

    print(f"  → {disp} | mood: {mood} | {weather.get('temp_c')}°C | {tc}")

    by = {}
    for f in all_features: by.setdefault(f["category"], []).append(f)
    def fok(f): return f.get("formality", 0.5) >= min_f

    ash = OCC_SHOE.get(disp)
    fw = [f for f in by.get("footwear", []) if
          (not ash or f.get("subcategory") in ash) and
          not (f.get("subcategory") == "sandals" and (layer in ["heavy", "extreme"] or tc == "morning")) and
          (fok(f) if min_f > 0 else True)] or by.get("footwear", [])
    seen_s, ufw = set(), []
    for f in fw:
        s = f.get("subcategory", "?")
        if s not in seen_s: seen_s.add(s); ufw.append(f)
    fw = ufw

    bp = OCC_BOTTOM.get(disp)
    bots = [f for f in by.get("bottom", []) if bottom_ok(f, disp, occ, layer, tc, bp, min_f)] or by.get("bottom", [])

    op = OCC_OUTER.get(disp); ao = by.get("outerwear", [])
    if op is not None:
        outer = [] if not op else ([f for f in ao if f.get("subcategory") in op] or ao)
    else:
        outer = [f for f in ao if fok(f)] if min_f > 0 else ao

    tops = [f for f in by.get("top", []) if fok(f)] if min_f > 0 else by.get("top", [])

    def fbok(f):
        s = f.get("subcategory", "?"); ok = FB_OK.get(s, ["casual"])
        return disp in ok or occ in ok
    fb = [f for f in by.get("full_body", []) if fbok(f)]

    cands, seen = [], set()
    def add(combo):
        k = frozenset(f["item_id"] for f in combo)
        if k not in seen: seen.add(k); cands.append(combo)

    for shoe in (fw or [None]):
        sl = [shoe] if shoe else []
        for t, b in itertools.product(tops, bots): add([t, b] + sl)
        for o, t, b in itertools.product(outer, tops, bots):
            if o.get("subcategory") == t.get("subcategory"): continue
            add([o, t, b] + sl)
        for f in fb: add([f] + sl)

    valid = []
    for combo in cands:
        if not outfit_complete([f["category"] for f in combo]): continue
        if not ctx_filter(combo, weather, occ, tc): continue
        ns = [f for f in combo if f["category"] != "footwear"]
        if not all(color_ok(f1["color_palette"], f2["color_palette"])[0]
                   for f1, f2 in itertools.combinations(ns, 2)): continue
        cs = clip_score(combo); fs = formality_score(combo)
        base = round(cs*0.65 + fs*0.35, 4)
        biased = apply_mood_score(base, combo, mood)        # ★ mood
        biased = apply_weather_score(biased, combo, weather) # ★ weather
        valid.append({
            "items": [f["item_id"] for f in combo],
            "labels": [f.get("subcategory", f["label"]) for f in combo],
            "categories": [f["category"] for f in combo],
            "fabrics": [f.get("fabric", "?") for f in combo],
            "formality": [f.get("formality", 0.5) for f in combo],
            "final_score": biased, "context": ctx,
        })

    if not valid:
        msgs = {"formal": "Formal kıyafet yetersiz.",
                "night": "Gece için yeterli parça bulunamadı."}
        return [{"items": [], "labels": [], "categories": [], "fabrics": [],
                 "formality": [], "final_score": 0, "style_axis": "",
                 "message": msgs.get(disp, "Bu his/event için uygun kombin bulunamadı."),
                 "context": ctx}]

    valid.sort(key=lambda x: x["final_score"], reverse=True)
    pool = valid[:25]; selected = []
    for c in pool:
        if len(selected) >= top_k: break
        ci = set(c["items"]); cl = set(c["labels"])
        if not any(len(ci & set(s["items"]))/len(ci | set(s["items"])) > 0.5 or
                   len(cl & set(s["labels"]))/len(cl | set(s["labels"])) > 0.75 for s in selected):
            c["style_axis"] = style_axis(c); selected.append(c)
    for o in pool:
        if len(selected) >= top_k: break
        if o not in selected: o["style_axis"] = style_axis(o); selected.append(o)
    return selected

# ── SESSION 2 (final): TEN RENGİ + AVATAR ──
def detect_skin_tone(selfie_path):
    """Selfie → fair | light | medium | brown | dark"""
    import cv2, mediapipe as mp
    img = cv2.imread(selfie_path)
    if img is None: return "medium"
    mp_face = mp.solutions.face_detection
    with mp_face.FaceDetection(min_detection_confidence=0.5) as fd:
        res = fd.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.detections: return "medium"
    bb = res.detections[0].location_data.relative_bounding_box
    h, w = img.shape[:2]
    x = max(0, int(bb.xmin*w)); y = max(0, int(bb.ymin*h))
    fw = int(bb.width*w); fh = int(bb.height*h)
    if fw < 50 or fh < 50: return "medium"
    face = img[y:y+fh, x:x+fw]
    cheek = face[int(fh*0.55):int(fh*0.78), int(fw*0.15):int(fw*0.85)]
    if cheek.size == 0: return "medium"
    ycrcb = cv2.cvtColor(cheek, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycrcb, np.array([0,133,77]), np.array([255,173,127]))
    if (skin > 0).sum() < 100: return "medium"
    y_ch = ycrcb[:,:,0]
    avg = float(np.mean(y_ch[skin > 0]))
    if avg > 200: return "fair"
    if avg > 170: return "light"
    if avg > 130: return "medium"
    if avg > 95:  return "brown"
    return "dark"


def select_avatar_for_user(gender, skin_tone):
    """gender + skin_tone → avatar dosya adı"""
    prefix = "avatar_m" if gender == "male" else "avatar_f"
    valid = {"fair","light","medium","brown","dark"}
    tone = skin_tone if skin_tone in valid else "medium"
    return f"{prefix}_{tone}.png"


def setup_user_profile(user_id, gender, selfie_path=None):
    """Kullanıcı kaydı: ten rengi tespit + avatar ata + DB güncelle"""
    tone = detect_skin_tone(selfie_path) if selfie_path else "medium"
    avatar_key = select_avatar_for_user(gender, tone)
    supabase.table("users").update({
        "gender": gender, "skin_tone": tone, "avatar_key": avatar_key
    }).eq("id", user_id).execute()
    print(f"  ✓ {gender} | {tone} | {avatar_key}")
    return {"gender": gender, "skin_tone": tone, "avatar_key": avatar_key}
