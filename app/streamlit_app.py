"""
Cricket Batting Shot Recognition - Streamlit demo
Advanced Deep Learning project, IIIT Sri City
Team: B Vishwesh (S20230010048), B Srimanreddy (S20230010049), E Tharun (S20230010081)

Pipeline (same as our training notebooks 04 and 06):
  video -> 16 uniformly sampled frames -> YOLO striker + bat detection -> square crop around the batter
  -> EfficientNet-B0 (fine-tuned) -> spatial attention -> 2-layer Bi-GRU -> temporal attention -> 15 shot classes

Model weights are downloaded from a Hugging Face model repository on first start (set HF_REPO below),
or loaded from MODEL_DIR if the three .pt files are already there (useful for running locally).
"""
import os
import tempfile
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

# ----------------------------------------------------------------------------- settings
HF_REPO = "Srimanreddy822/cricket-shot-models"   
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
WEIGHT_FILES = ["best_finetuned.pt", "Player_Type_Detection_Model.pt", "Bat_Detection_Model.pt"]

NUM_FRAMES, IMG = 16, 224          # must match training
PAD, CONF, YOLO_IMGSZ = 0.10, 0.25, 640
YOLO_CHUNK = 4                     # frames per YOLO call (keeps memory low on the free tier)
MAX_SECONDS = 20                   # only the first 20 s of a long upload are read
CLASSES = ["Cover Drive", "Defensive", "Down The Wicket", "Flick", "Hook", "Late Cut",
           "Lofted Legside", "Lofted Offside", "Pull", "Reverse Sweep", "Scoop",
           "Square Cut", "Straight Drive", "Sweep", "Upper Cut"]   # label order used in training
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


def attention_plot(attn, pred_name):
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.bar(np.arange(1, NUM_FRAMES + 1), attn * 100, color="#2b6cb0")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Attention (%)")
    ax.set_xticks(np.arange(1, NUM_FRAMES + 1))
    ax.set_title(f"Which frames the model used to predict '{pred_name}'")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig

# ----------------------------------------------------------------------------- UI
st.set_page_config(page_title="Cricket Shot Recognition", page_icon="🏏", layout="wide")
st.title("🏏 Cricket Batting Shot Recognition")
st.markdown(
    "Upload a short **broadcast-style clip** (camera behind the bowler, about 1–2 seconds, one shot). "
    "The model predicts which of **15 batting shots** was played and shows **which frames it focused on**.\n\n"
    "**How it works:** 16 frames are sampled → YOLO finds the striker and bat → each frame is cropped around the "
    "batter → a fine-tuned **EfficientNet-B0** describes each frame → a **bidirectional GRU with temporal "
    "attention** reads the sequence and classifies the shot. Test accuracy on unseen matches: **82.9%** "
    "(top-3: 92.6%)."
)

model, striker, bat, striker_id, bat_id = load_models()

left, right = st.columns([1, 1])
with left:
    upload = st.file_uploader("Batting clip", type=["mp4", "avi", "mov", "mkv", "webm"])
    c1, c2 = st.columns(2)
    start = c1.number_input("Start time (s)", min_value=0.0, value=0.0, step=0.5)
    end = c2.number_input("End time (s), 0 = to the end", min_value=0.0, value=0.0, step=0.5)
    run = st.button("Recognise shot", type="primary", disabled=upload is None)
    if upload is not None:
        st.video(upload)

if run and upload is not None:
    suffix = os.path.splitext(upload.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.getbuffer())
        path = tmp.name
    t0 = time.time()
    with st.spinner("Detecting the batter and classifying the shot..."):
        frames, duration = read_frames(path, start, end)
        if frames is None:
            st.error("Could not read any frames. Check the file and the start/end times.")
            st.stop()
        s_cands, b_boxes = detect(frames, striker, bat, striker_id, bat_id)
        crops, boxes, found = batter_crops(frames, s_cands, b_boxes)
        probs, attn = classify(model, crops)
    os.remove(path)

    order = probs.argsort()[::-1]
    pred = CLASSES[order[0]]
    with right:
        st.subheader(f"Prediction: {pred}")
        st.metric("Confidence", f"{probs[order[0]] * 100:.1f}%")
        for i in order[:3]:
            st.progress(float(probs[i]), text=f"{CLASSES[i]}: {probs[i] * 100:.1f}%")
        st.caption(f"Processed in {time.time() - t0:.1f} s · clip length used {duration:.1f} s · "
                   f"batter confirmed in {int(found.sum())}/{NUM_FRAMES} frames")
        if duration > 3:
            st.warning("The model was trained on ~1 s clips containing one shot. "
                       "For long videos, set the start and end times around the shot.")
        if found.sum() == 0:
            st.warning("The batter was not confidently detected, so this prediction may be unreliable.")

    st.pyplot(attention_plot(attn, pred))

    st.subheader("What the model saw")
    st.caption("Cropped frames fed to the model, with each frame's attention weight")
    for row in range(0, NUM_FRAMES, 8):
        cols = st.columns(8)
        for c, t in zip(cols, range(row, row + 8)):
            c.image(crops[t], caption=f"frame {t + 1} · {attn[t] * 100:.0f}%")

    st.caption("Batter detection on the original frames (yellow = crop region)")
    cols = st.columns(4)
    for c, t in zip(cols, [0, 5, 10, 15]):
        img = cv2.cvtColor(frames[t], cv2.COLOR_BGR2RGB).copy()
        x1, y1, x2, y2 = boxes[t].astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 210, 0), max(2, img.shape[1] // 250))
        c.image(img, caption=f"frame {t + 1}" + (" · batter found" if found[t] else " · fallback box"))

st.divider()
st.markdown(
    "**Limitations.** Trained only on broadcast footage filmed from behind the bowler. Side-on, phone or "
    "net-practice videos may give wrong answers. The most common confusions are Hook ↔ Pull, "
    "Lofted Offside ↔ Lofted Legside and Straight Drive ↔ Cover Drive. Runs on a free CPU, so expect "
    "roughly 10–40 seconds per clip.\n\n"
    "**Credits.** Dataset and striker/bat detection models: Dihan et al., *CricShot10k: A Large-Scale Video "
    "Dataset for Cricket Shot Classification*, IEEE Access, 2026 "
    "([GitHub](https://github.com/Dihan69/CricShot10k)), released for non-commercial research use. "
    "Model architecture follows *Modern Deep Learning Approaches for Cricket Shot Classification: "
    "A Comprehensive Baseline Study*, arXiv:2510.09187, 2025. "
    "Non-commercial student project (Advanced Deep Learning, IIIT Sri City): B Vishwesh, B Srimanreddy, E Tharun."
)