"""
Which shot was that? - Cricket batting shot recognition (Streamlit app)
Advanced Deep Learning project, IIIT Sri City
Team: B Vishwesh (S20230010048), B Srimanreddy (S20230010049), E Tharun (S20230010081)

Pipeline (same as our training notebooks 04 and 06):
  video -> 16 uniformly sampled frames -> YOLO striker + bat detection -> square crop around the batter
  -> EfficientNet-B0 (fine-tuned) -> spatial attention -> 2-layer Bi-GRU -> temporal attention -> 15 shot classes

Model weights are downloaded from a Hugging Face model repository on first start (set HF_REPO below),
or loaded from MODEL_DIR if the three .pt files are already there (useful for running locally).
"""
import base64
import html
import os
import tempfile
import time

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

# ----------------------------------------------------------------------------- settings
HF_REPO = "Srimanreddy822/cricket-shot-models"  
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

NUM_FRAMES, IMG = 16, 224          # must match training
PAD, CONF, YOLO_IMGSZ = 0.10, 0.25, 640
YOLO_CHUNK = 4                     # frames per YOLO call (keeps memory low on the free tier)
MAX_SECONDS = 20                   # only the first 20 s of a long upload are read
CLASSES = ["Cover Drive", "Defensive", "Down The Wicket", "Flick", "Hook", "Late Cut",
           "Lofted Legside", "Lofted Offside", "Pull", "Reverse Sweep", "Scoop",
           "Square Cut", "Straight Drive", "Sweep", "Upper Cut"]   # label order used in training

SHOT_NOTES = {
    "Cover Drive": "A front-foot drive along the ground through the covers, on the off side.",
    "Defensive": "A blocking stroke played to keep the ball out rather than to score.",
    "Down The Wicket": "The batter steps out of the crease towards the bowler to meet the ball early.",
    "Flick": "A wristy stroke that turns a straight ball away towards the leg side.",
    "Hook": "A horizontal-bat shot to a short ball around head height, hit to the leg side.",
    "Late Cut": "A delicate cut played late, steering the ball fine past the slips.",
    "Lofted Legside": "A shot hit in the air over the leg-side field.",
    "Lofted Offside": "A shot hit in the air over the off-side field.",
    "Pull": "A horizontal-bat shot to a short ball around waist height, hit to the leg side.",
    "Reverse Sweep": "A sweep with the hands reversed, sending the ball to the off side.",
    "Scoop": "The ball is lifted over the wicketkeeper with an open bat face.",
    "Square Cut": "A horizontal-bat shot to a short, wide ball, hit square on the off side.",
    "Straight Drive": "A front-foot drive played straight back past the bowler.",
    "Sweep": "Played down on one knee, sweeping the ball across the line to the leg side.",
    "Upper Cut": "A short, wide ball steered upwards over the slips or third man.",
}
torch.set_num_threads(max(1, os.cpu_count() or 2))

# ----------------------------------------------------------------------------- model (identical to notebook 06)
class SpatialAttention(nn.Module):
    """Squeeze-and-excitation style channel re-weighting of the 1280-d frame features."""
    def __init__(self, d):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(d, d // 8), nn.ReLU(), nn.Linear(d // 8, d), nn.Sigmoid())

    def forward(self, x):
        return x * self.att(x)


class TemporalAttentionPaper(nn.Module):
    """Scores every frame, softmax over time, weighted sum of the GRU outputs."""
    def __init__(self, d):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d, d // 4), nn.ReLU(), nn.Linear(d // 4, 1))

    def forward(self, H):
        a = torch.softmax(self.score(H).squeeze(-1), dim=1)
        return (a.unsqueeze(-1) * H).sum(1), a


class PaperModel(nn.Module):
    def __init__(self, num_classes, in_dim=1280, hidden=256):
        super().__init__()
        self.se = SpatialAttention(in_dim)
        self.rnn = nn.GRU(in_dim, hidden, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.attn = TemporalAttentionPaper(2 * hidden)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * hidden, 512), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Linear(512, num_classes))

    def forward(self, x):
        z, a = self.attn(self.rnn(self.se(x))[0])
        return self.head(z), a


class FullModel(nn.Module):
    """EfficientNet-B0 applied to every frame, followed by the temporal model."""
    def __init__(self, num_classes):
        super().__init__()
        b0 = efficientnet_b0(weights=None)                      # weights come from our checkpoint
        self.backbone = nn.Sequential(b0.features, b0.avgpool, nn.Flatten())
        self.temporal = PaperModel(num_classes)

    def forward(self, x):                                       # x: (B, T, 3, H, W)
        B, T = x.shape[:2]
        f = self.backbone(x.flatten(0, 1)).view(B, T, -1)
        return self.temporal(f)


def weight_path(name):
    local = os.path.join(MODEL_DIR, name)
    if os.path.exists(local):
        return local
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=HF_REPO, filename=name)


@st.cache_resource(show_spinner="Loading models (first start can take a minute)...")
def load_models():
    from ultralytics import YOLO
    model = FullModel(len(CLASSES))
    state = torch.load(weight_path("best_finetuned.pt"), map_location="cpu")
    model.load_state_dict(state.get("state_dict", state) if isinstance(state, dict) else state)
    model.eval()
    striker = YOLO(weight_path("Player_Type_Detection_Model.pt"))
    bat = YOLO(weight_path("Bat_Detection_Model.pt"))
    striker_id = [k for k, v in striker.names.items() if v.lower() == "striker"][0]
    bat_id = [k for k, v in bat.names.items() if "bat" in v.lower()][0]
    return model, striker, bat, striker_id, bat_id


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# ----------------------------------------------------------------------------- video + cropping (as in notebook 04)
def read_frames(path, start, end):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    s = int(max(0.0, start or 0.0) * fps)
    e = int(end * fps) if end and end > (start or 0) else int(MAX_SECONDS * fps) + s
    frames, i = [], 0
    while i < e:
        ok, f = cap.read()
        if not ok:
            break
        if i >= s:
            frames.append(f)
        i += 1
    cap.release()
    if not frames:
        return None, 0.0
    idx = np.linspace(0, len(frames) - 1, NUM_FRAMES).round().astype(int)
    return [frames[j] for j in idx], len(frames) / fps


def union(a, b):
    return np.array([min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])])


def square_box(box, W, H, pad=PAD, min_side=96):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = int(min(max(max(x2 - x1, y2 - y1) * (1 + 2 * pad), min_side), W, H))
    x1 = int(np.clip(round(cx - side / 2), 0, W - side))
    y1 = int(np.clip(round(cy - side / 2), 0, H - side))
    return np.array([x1, y1, x1 + side, y1 + side])


def crop_resize(frame_bgr, box):
    x1, y1, x2, y2 = box.astype(int)
    crop = cv2.resize(frame_bgr[y1:y2, x1:x2], (IMG, IMG), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def bat_inside(bat, box, grow=0.3):
    bx, by = (bat[0] + bat[2]) / 2, (bat[1] + bat[3]) / 2
    w, h = box[2] - box[0], box[3] - box[1]
    return (box[0] - grow * w <= bx <= box[2] + grow * w) and (box[1] - grow * h <= by <= box[3] + grow * h)


def detect(frames, striker, bat, striker_id, bat_id):
    s_cands, b_boxes = [], []
    for s in range(0, len(frames), YOLO_CHUNK):
        chunk = frames[s:s + YOLO_CHUNK]
        rs = striker.predict(chunk, imgsz=YOLO_IMGSZ, conf=CONF, classes=[striker_id], verbose=False)
        rb = bat.predict(chunk, imgsz=YOLO_IMGSZ, conf=CONF, classes=[bat_id], max_det=1, verbose=False)
        for a, b in zip(rs, rb):
            s_cands.append(np.concatenate([a.boxes.xyxy.numpy(), a.boxes.conf.numpy()[:, None]], 1)
                           if len(a.boxes) else np.zeros((0, 5)))
            b_boxes.append(b.boxes.xyxy[0].numpy() if len(b.boxes) else None)
    return s_cands, b_boxes


def choose_striker(s_cands, b_boxes, W):
    """Trust the striker candidate that holds the detected bat; drop outliers; fall back to most confident."""
    T = len(s_cands)
    chosen = [None] * T
    for t in range(T):
        if b_boxes[t] is None or len(s_cands[t]) == 0:
            continue
        holding = [c for c in s_cands[t] if bat_inside(b_boxes[t], c)]
        if holding:
            chosen[t] = max(holding, key=lambda c: c[4])[:4]
    idx = [t for t in range(T) if chosen[t] is not None]
    if len(idx) >= 3:
        cx = np.array([(chosen[t][0] + chosen[t][2]) / 2 for t in idx])
        med = np.median(cx)
        for t, c in zip(idx, cx):
            if abs(c - med) > 0.25 * W:
                chosen[t] = None
    if all(c is None for c in chosen):
        chosen = [c[c[:, 4].argmax(), :4] if len(c) else None for c in s_cands]
    return chosen


def batter_crops(frames, s_cands, b_boxes):
    H, W = frames[0].shape[:2]
    chosen = choose_striker(s_cands, b_boxes, W)
    raw = []
    for t, c in enumerate(chosen):
        if c is None:
            raw.append(None)
        elif b_boxes[t] is not None and bat_inside(b_boxes[t], c):
            raw.append(union(c, b_boxes[t]))
        else:
            raw.append(c)
    found = np.array([r is not None for r in raw])
    if found.any():
        rs = np.stack([r for r in raw if r is not None])
        clip_box = square_box(np.array([rs[:, 0].min(), rs[:, 1].min(), rs[:, 2].max(), rs[:, 3].max()]), W, H)
    else:
        clip_box = np.array([0, 0, W, H])
    boxes = [square_box(r, W, H) if r is not None else clip_box for r in raw]
    crops = np.stack([crop_resize(f, b) for f, b in zip(frames, boxes)])
    return crops, boxes, found


@torch.no_grad()
def classify(model, crops):
    x = torch.from_numpy(crops).permute(0, 3, 1, 2).float().div(255.0)
    x = ((x - MEAN) / STD).unsqueeze(0)                         # (1, 16, 3, 224, 224)
    logits, attn = model(x)
    return torch.softmax(logits, dim=1)[0].numpy(), attn[0].numpy()


# ----------------------------------------------------------------------------- presentation helpers
def to_data_uri(rgb, quality=82):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def run_pipeline(path, start, end):
    model, striker, bat, striker_id, bat_id = load_models()
    t0 = time.time()
    frames, duration = read_frames(path, start, end)
    if frames is None:
        return None
    s_cands, b_boxes = detect(frames, striker, bat, striker_id, bat_id)
    crops, boxes, found = batter_crops(frames, s_cands, b_boxes)
    probs, attn = classify(model, crops)
    detections = []
    for t in [0, 5, 10, 15]:
        img = cv2.cvtColor(frames[t], cv2.COLOR_BGR2RGB).copy()
        x1, y1, x2, y2 = boxes[t].astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (31, 111, 74), max(3, img.shape[1] // 200))
        scale = 640 / img.shape[1]
        detections.append((cv2.resize(img, None, fx=scale, fy=scale), t + 1, bool(found[t])))
    return {"probs": probs, "attn": attn, "crops": crops, "found": found, "detections": detections,
            "duration": duration, "seconds": time.time() - t0}


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
  --bg: #F7F8F7; --surface: #FFFFFF; --ink: #18201B; --muted: #5E6B63;
  --turf: #1F6F4A; --turf-tint: #E4F0E9; --line: #E3E7E4;
}
html, body, .stApp, .stMarkdown, p, li, label, button, input, textarea, h1, h2, h3, h4 {
  font-family: 'Manrope', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.stApp { background: var(--bg); color: var(--ink); }
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stDecoration"] { display: none; }
.block-container { max-width: 1040px; padding-top: 4.5rem; padding-bottom: 4rem; }

/* Hero */
.hero h1 {
  font-size: clamp(2.6rem, 6vw, 4.6rem); font-weight: 800; letter-spacing: -0.035em;
  line-height: 1.02; margin: 0 0 1.1rem 0; color: var(--ink);
}
.hero p { font-size: 1.2rem; line-height: 1.55; color: var(--muted); max-width: 36rem; margin: 0; }

/* Section headings */
.section-title { font-size: 1.5rem; font-weight: 700; letter-spacing: -0.02em; margin: 3.5rem 0 0.4rem 0; }
.section-note { color: var(--muted); font-size: 1rem; line-height: 1.55; margin: 0 0 1.2rem 0; max-width: 40rem; }

/* Upload panel */
[data-testid="stFileUploaderDropzone"] {
  background: var(--surface); border: 1.5px dashed #C9D3CD; border-radius: 18px; padding: 1.6rem;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--turf); }
.stButton > button, [data-testid="stBaseButton-primary"] {
  border-radius: 999px; padding: 0.7rem 1.6rem; font-weight: 700; font-size: 1rem;
  background: var(--turf); border: none; color: #fff;
}
.stButton > button:hover { background: #185C3D; color: #fff; }
.stButton > button:focus-visible { outline: 3px solid #9CC9B0; outline-offset: 2px; }
[data-testid="stExpander"] details { border: 1px solid var(--line); border-radius: 14px; background: var(--surface); }

/* Result */
.result { margin-top: 3.5rem; }
.result .lead { color: var(--muted); font-size: 1.05rem; margin: 0 0 0.3rem 0; }
.result .shot {
  font-size: clamp(3rem, 9vw, 6.2rem); font-weight: 800; letter-spacing: -0.045em; line-height: 0.98;
  margin: 0 0 1rem 0; color: var(--ink);
}
.result .about { font-size: 1.15rem; line-height: 1.55; color: var(--ink); max-width: 34rem; margin: 0 0 0.6rem 0; }
.result .confidence { font-size: 1.05rem; color: var(--turf); font-weight: 700; margin: 0; }
.result .meta { color: var(--muted); font-size: 0.92rem; margin-top: 1.2rem; }

.ranking { background: var(--surface); border: 1px solid var(--line); border-radius: 20px; padding: 1.4rem 1.5rem; margin-top: 0.4rem; }
.ranking h3 { font-size: 1rem; font-weight: 700; margin: 0 0 1rem 0; }
.rank-row { margin-bottom: 1rem; }
.rank-row:last-child { margin-bottom: 0; }
.rank-label { display: flex; justify-content: space-between; font-size: 0.98rem; margin-bottom: 0.35rem; }
.rank-label span:last-child { color: var(--muted); font-variant-numeric: tabular-nums; }
.rank-track { height: 8px; background: var(--turf-tint); border-radius: 999px; overflow: hidden; }
.rank-fill { height: 100%; background: var(--turf); border-radius: 999px; }
.rank-row.other .rank-fill { background: #9CB5A6; }

.notice { margin-top: 1.2rem; padding: 0.9rem 1.1rem; border-radius: 14px; background: #FFF6E5; color: #6A4A06; font-size: 0.95rem; line-height: 1.5; }

/* Film strip: frames with attention bars */
.strip { display: grid; grid-template-columns: repeat(16, minmax(0, 1fr)); gap: 6px; align-items: end; }
.frame { display: flex; flex-direction: column; align-items: stretch; }
.bar-zone { height: 84px; display: flex; align-items: flex-end; margin-bottom: 6px; }
.bar { width: 100%; background: #C3DCCD; border-radius: 6px 6px 2px 2px; transform-origin: bottom;
       animation: grow 900ms cubic-bezier(.2,.8,.2,1) both; }
.frame.peak .bar { background: var(--turf); }
.frame img { width: 100%; aspect-ratio: 1 / 1; object-fit: cover; border-radius: 6px; display: block; }
.frame.peak img { outline: 2.5px solid var(--turf); outline-offset: 1px; }
.frame small { text-align: center; color: var(--muted); font-size: 0.72rem; margin-top: 4px; font-variant-numeric: tabular-nums; }
@keyframes grow { from { transform: scaleY(0); } to { transform: scaleY(1); } }
@media (prefers-reduced-motion: reduce) { .bar { animation: none; } }
@media (max-width: 760px) {
  .strip { grid-template-columns: repeat(8, minmax(0, 1fr)); row-gap: 14px; }
  .block-container { padding-top: 2.5rem; }
}

.detections { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
.detections figure { margin: 0; }
.detections img { width: 100%; border-radius: 10px; display: block; }
.detections figcaption { color: var(--muted); font-size: 0.82rem; margin-top: 0.35rem; }
@media (max-width: 760px) { .detections { grid-template-columns: repeat(2, minmax(0, 1fr)); } }

/* How it works */
.steps { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1.6rem; margin-top: 0.4rem; }
.step b { display: block; font-size: 1.02rem; margin-bottom: 0.35rem; }
.step b span { color: var(--turf); margin-right: 0.35rem; }
.step p { color: var(--muted); font-size: 0.95rem; line-height: 1.55; margin: 0; }
@media (max-width: 760px) { .steps { grid-template-columns: 1fr; gap: 1.1rem; } }

.site-footer { margin-top: 4rem; padding-top: 1.6rem; border-top: 1px solid var(--line); color: var(--muted); font-size: 0.88rem; line-height: 1.6; }
.site-footer p { margin: 0 0 0.7rem 0; max-width: 46rem; }
.site-footer a { color: var(--turf); }
.empty { margin-top: 2.2rem; color: var(--muted); font-size: 1rem; }
</style>
"""


def html_block(markup):
    st.markdown(markup, unsafe_allow_html=True)


def render_result(res):
    probs, attn = res["probs"], res["attn"]
    order = probs.argsort()[::-1]
    pred = CLASSES[order[0]]
    conf = probs[order[0]] * 100

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        html_block(
            f'<div class="result"><p class="lead">The model thinks this is a</p>'
            f'<div class="shot">{html.escape(pred)}</div>'
            f'<p class="about">{html.escape(SHOT_NOTES[pred])}</p>'
            f'<p class="confidence">{conf:.0f}% confident</p>'
            f'<p class="meta">Analysed {res["duration"]:.1f} s of video in {res["seconds"]:.1f} s. '
            f'The batter was confirmed in {int(res["found"].sum())} of {NUM_FRAMES} frames.</p></div>'
        )
    with right:
        rows = []
        for rank, i in enumerate(order[:3]):
            rows.append(
                f'<div class="rank-row{"" if rank == 0 else " other"}"><div class="rank-label">'
                f'<span>{html.escape(CLASSES[i])}</span><span>{probs[i] * 100:.1f}%</span></div>'
                f'<div class="rank-track"><div class="rank-fill" style="width:{max(probs[i] * 100, 1.5):.1f}%">'
                f'</div></div></div>'
            )
        html_block('<div class="result"><div class="ranking"><h3>Closest matches</h3>' + "".join(rows) + "</div></div>")
        notes = []
        if res["duration"] > 3:
            notes.append("This clip is longer than the one-second clips the model was trained on. "
                         "Set start and end times around the shot for a more reliable answer.")
        if res["found"].sum() == 0:
            notes.append("The batter could not be confirmed in any frame, so treat this answer with caution.")
        if notes:
            html_block('<div class="notice">' + " ".join(notes) + "</div>")

    peak = int(attn.argmax())
    cells = []
    for t in range(NUM_FRAMES):
        height = max(attn[t] / attn.max() * 100, 4)
        cells.append(
            f'<div class="frame{" peak" if t == peak else ""}">'
            f'<div class="bar-zone"><div class="bar" style="height:{height:.0f}%; animation-delay:{t * 35}ms"></div></div>'
            f'<img src="{to_data_uri(res["crops"][t])}" alt="Cropped frame {t + 1}">'
            f'<small>{t + 1}</small></div>'
        )
    html_block('<div class="section-title">What the model focused on</div>'
               '<p class="section-note">These are the 16 frames the model saw, cropped around the batter. '
               'Taller bars mark the frames it relied on most; the darkest one is the moment that mattered most.</p>'
               '<div class="strip">' + "".join(cells) + "</div>")

    with st.expander("See how the batter was found in the original video"):
        figs = "".join(
            f'<figure><img src="{to_data_uri(img)}" alt="Frame {n} with batter box">'
            f'<figcaption>Frame {n}: {"batter confirmed holding the bat" if ok else "estimated from nearby frames"}'
            f'</figcaption></figure>'
            for img, n, ok in res["detections"]
        )
        html_block('<div class="detections">' + figs + "</div>")


# ----------------------------------------------------------------------------- page
st.set_page_config(page_title="Which shot was that?", page_icon="🏏", layout="wide")
html_block(CSS)

html_block(
    '<div class="hero"><h1>Which shot was that?</h1>'
    "<p>Upload a broadcast clip of a single delivery. The model follows the batter's movement "
    "frame by frame and names the shot, from a cover drive to a reverse sweep.</p></div>"
)

html_block('<div class="section-title">Your clip</div>'
           '<p class="section-note">Use the usual TV view from behind the bowler, with one shot lasting about '
           'a second. MP4, AVI, MOV and WebM files up to 50 MB work.</p>')
upload = st.file_uploader("Your clip", type=["mp4", "avi", "mov", "mkv", "webm"], label_visibility="collapsed")

with st.expander("Trim a longer video"):
    c1, c2 = st.columns(2)
    start = c1.number_input("Start time in seconds", min_value=0.0, value=0.0, step=0.5)
    end = c2.number_input("End time in seconds (0 means the end of the video)", min_value=0.0, value=0.0, step=0.5)

if upload is not None and os.path.splitext(upload.name)[1].lower() in (".mp4", ".webm", ".mov"):
    st.video(upload)

load_models()   # cached: slow only on the very first start of the app
identify = st.button("Identify shot", type="primary", disabled=upload is None)

if identify and upload is not None:
    suffix = os.path.splitext(upload.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.getbuffer())
        path = tmp.name
    try:
        with st.spinner("Finding the batter and reading the shot. This takes 10 to 40 seconds."):
            result = run_pipeline(path, start, end)
    finally:
        os.remove(path)
    if result is None:
        st.error("No frames could be read from this file. Check that it is a video and that the start and "
                 "end times fall inside it.")
        st.session_state.pop("result", None)
    else:
        st.session_state["result"] = result
        st.session_state["result_for"] = upload.name

if upload is None:
    st.session_state.pop("result", None)

if "result" in st.session_state and upload is not None and st.session_state.get("result_for") == upload.name:
    render_result(st.session_state["result"])
else:
    html_block('<p class="empty">Upload a clip and select Identify shot to see the result here.</p>')

html_block(
    '<div class="section-title">How it works</div>'
    '<p class="section-note">Each clip goes through the same four stages the model was trained with.</p>'
    '<div class="steps">'
    '<div class="step"><b><span>1</span>Sample the clip</b><p>Sixteen evenly spaced frames are taken from '
    'start to finish, so every clip has the same length.</p></div>'
    '<div class="step"><b><span>2</span>Find the batter</b><p>Two detectors locate the striker and the bat, '
    'and each frame is cropped to a square around them.</p></div>'
    '<div class="step"><b><span>3</span>Read each frame</b><p>A fine-tuned EfficientNet-B0 turns every crop '
    'into a description of the body position and bat angle.</p></div>'
    '<div class="step"><b><span>4</span>Follow the motion</b><p>A bidirectional GRU reads the frames in '
    'order, and attention picks out the moments that decide the shot.</p></div>'
    "</div>"
)

html_block(
    '<div class="site-footer">'
    "<p>On 1,350 clips from 76 matches the model never saw during training, it named the right shot 82.9% of "
    "the time, and the right shot was among its top three 92.6% of the time. It was trained only on broadcast "
    "footage filmed from behind the bowler, so side-on, phone or net-practice videos may be misread. Hook and "
    "pull, lofted shots to either side, and straight and cover drives are the pairs it confuses most.</p>"
    '<p>Dataset and batter detection models by Dihan et al., '
    '<a href="https://github.com/Dihan69/CricShot10k" target="_blank" rel="noopener">CricShot10k</a> '
    "(IEEE Access, 2026), released for non-commercial research. Model design follows the baseline study "
    "<em>Modern Deep Learning Approaches for Cricket Shot Classification</em> (arXiv:2510.09187).</p>"
    "<p>A student project for Advanced Deep Learning at IIIT Sri City by B Vishwesh, B Srimanreddy and "
    "E Tharun.</p></div>"
)