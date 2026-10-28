# Cricket Batting Shot Recognition from Video

Recognising which of **15 cricket batting shots** is played in a short broadcast clip, using a CNN (EfficientNet-B0) to understand each frame and a recurrent network (Bi-GRU with temporal attention) to understand the motion over time.

---

## Results at a glance

All results are on a **held-out test set of 76 unseen matches** (1,350 clips). The split is made **by match**, so no match appears in more than one of train, validation or test.

| Setup | Test accuracy | Macro F1 | Top-3 |
|---|---|---|---|
| Mean-pool over frames, no RNN (frozen B0, full frame) | 45.3% | 0.447 | 76.2% |
| Bi-GRU + attention (frozen B0, full frame) | 64.4% | 0.628 | 85.2% |
| + batter cropping (frozen B0, per-frame crop) | 72.0% | 0.712 | 87.4% |
| Baseline paper's head and training recipe (frozen B0, crop) | 72.2% | 0.713 | 87.7% |
| **Baseline paper's model, fine-tuned end to end (B0, crop)** | **82.9%** | **0.828** | **92.6%** |

Frozen-backbone results are the mean of 3 seeds; the fine-tuned result is a single run.

**What drives accuracy:** temporal modelling (+19.1 points over mean pooling), cropping around the batter (+7.6), and fine-tuning the CNN end to end (+10.7). Matching the baseline paper's exact head and schedule on frozen features changed accuracy by less than the seed variance.

### Temporal model comparison (frozen B0, full frame, mean of 3 seeds)

| Model | Params | Test acc | Macro F1 |
|---|---|---|---|
| Mean-pool (no RNN) | 0.33M | 45.3 ± 0.5 | 0.447 |
| RNN | 0.60M | 58.1 ± 0.7 | 0.565 |
| LSTM | 2.17M | 61.2 ± 1.0 | 0.600 |
| GRU | 1.65M | 62.1 ± 0.5 | 0.608 |
| Bi-GRU | 3.68M | 60.4 ± 0.9 | 0.593 |
| RNN + attention | 0.63M | 61.7 ± 0.1 | 0.608 |
| LSTM + attention | 2.21M | 62.4 ± 1.4 | 0.618 |
| GRU + attention | 1.68M | 62.8 ± 1.7 | 0.615 |
| **Bi-GRU + attention** | 3.75M | **64.4 ± 0.3** | **0.628** |

### Input comparison (frozen B0, test accuracy, mean of 3 seeds)

| Model | Full frame | Per-frame crop | Per-clip crop |
|---|---|---|---|
| Mean-pool | 45.3 | 55.0 | 52.0 |
| LSTM + attn | 62.4 | 68.9 | 66.8 |
| GRU + attn | 62.8 | 69.6 | 67.8 |
| Bi-GRU + attn | 64.4 | **72.0** | 68.3 |

### Data leakage check (Bi-GRU + attn, frozen B0, full frame)

| Split | Test acc | Macro F1 |
|---|---|---|
| By match (used throughout) | 64.4 ± 0.3 | 0.628 |
| Random by clip | 64.9 ± 0.7 | 0.638 |

Splitting by match changed accuracy by only 0.5 points on CricShot10k, so match identity is not a major source of leakage in this dataset.

---

## Pipeline

```
Video clip (~1 s, 896×540)
  → sample 16 frames uniformly
  → detect striker + bat (YOLO models released by the CricShot10k authors)
  → square crop around the batter, resize to 224×224
  → EfficientNet-B0 (ImageNet-pretrained) → 1,280 features per frame
  → spatial attention (channel re-weighting)
  → 2-layer bidirectional GRU (256 hidden per direction)
  → temporal attention (weights the most informative frames)
  → classifier → 15 shot probabilities
```

**Batter cropping.** For each frame we keep the striker candidate whose box contains the detected bat; frames without such a match fall back to a box covering the trusted detections across the clip. Crops are made square before resizing so the batter is not distorted.

**Training (fine-tuned model).** AdamW (backbone lr 1e-4, rest 1e-3, weight decay 1e-4), StepLR (×0.5 every 10 epochs), batch 8 with 4-step gradient accumulation, mixed precision, gradient clipping 1.0, cross-entropy with label smoothing 0.1 and inverse-frequency class weights. Augmentation is applied **identically to all frames of a clip**: horizontal flip, ±10° rotation and colour jitter. Early stopping on validation loss (patience 10); the checkpoint with the best validation accuracy is kept.

---

## Repository structure

```
notebooks/
  01-download-cricshot10k.ipynb      Download the dataset archives
  02-split-and-features.ipynb        Metadata, match-level split, frozen B0 features (full frame)
  03-train-temporal-models.ipynb     RNN / LSTM / GRU / Bi-GRU ± attention, leakage test, crop comparison
  04-batter-crop-features.ipynb      Striker + bat detection, square crops, frozen B0 features
  05-paper-architecture-frozen.ipynb Baseline paper's head and training recipe on frozen features
  06-finetune-paper-model.ipynb      Baseline paper's model fine-tuned end to end
results/                             CSV result tables, confusion matrices, attention plots, training history
requirements.txt
```

The notebooks were run on **Kaggle** (GPU T4). Each one lists its inputs at the top; later notebooks use earlier notebooks' outputs.

## How to reproduce

1. Run notebook 01 on Kaggle with Internet on, and save the version. Its output holds the dataset archives.
2. Run 02 → 03 → 04 → 05 → 06 in order. For each, attach the outputs of the notebooks it needs (listed in the notebook) and use **Save Version → Save & Run All**.
3. Each notebook has a `DEBUG` flag. Run with `DEBUG = True` first to check everything works on a small subset.

Approximate run times on one T4: features 7–10 min (02), crop detection ~2 h (04), fine-tuning ~1.6 h (06).

---

## Dataset

**CricShot10k** (Dihan et al., IEEE Access 2026): 10,091 clips (as downloaded), 15 shot classes, from 534 match highlight videos. The dataset is for **non-commercial research use only**, and the original video copyright remains with the broadcasters. **No videos are included in this repository**; download them from the [official repository](https://github.com/Dihan69/CricShot10k).

## Acknowledgements and references

- **Model architecture and training recipe** follow the baseline study and its code: *Modern Deep Learning Approaches for Cricket Shot Classification: A Comprehensive Baseline Study*, arXiv:2510.09187, 2025 ([code](https://github.com/hpicsk/CricShot10_Baselines)). We reimplemented it, adapted it to CricShot10k (15 classes, 16 frames, match-level split) and made augmentation consistent across the frames of a clip.
- **Dataset, cropping approach and striker/bat detectors** come from M. K. Dihan, Abdullah, Amina and S. Ahmed, *CricShot10k: A Large-Scale Video Dataset for Cricket Shot Classification*, IEEE Access, vol. 14, pp. 23428–23447, 2026 ([code](https://github.com/Dihan69/CricShot10k)).
- M. Tan and Q. Le, *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*, ICML 2019.
- K. Cho et al., *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation*, EMNLP 2014 (GRU).
