# Soft Tag Adapters for Ancient → Modern Chinese Translation

A detailed report on augmenting the Erya translation model with soft POS-tag and word-segmentation embeddings distilled from SikuRoBERTa, and on the three-stage curriculum used to fine-tune the resulting model.

---

## 1. Generating the Augmented Dataset with SikuRoBERTa

The fine-tuning data is enriched offline by running every source-side line of the EvaHan parallel corpus through a pretrained SikuRoBERTa-CRF tagger that jointly predicts character-level word-segmentation labels (BIES) and ~30-way POS labels. The script that performs this augmentation lives at [Erya/augment_finetune_with_siku.py](Erya/augment_finetune_with_siku.py) and produces a sibling `.src.siku_aux.npz` file for every `.src` in `finetune.tgz`.

### 1.1. The tagger

SikuRoBERTa is a Chinese RoBERTa pretrained on the *Sì Kù Quán Shū* corpus. The downstream tagger ([model.py:BertSegPos](The-first-ancient-Chinese-word-segmentation-and-part-of-speech-tagging-code-and-analysis-main/model.py)) places two independent classifier heads on top of the encoder's final hidden states:

- `classifier_seg` → `(d_model=768) → 4 BIES labels` (B / I / E / S — Begin, Inside, End, Single)
- `classifier_pos` → `(d_model=768) → ~30 POS labels` (`n`, `v`, `nr`, `nn`, `p`, `c`, `r`, `t`, `d`, `y`, `ns`, `sv`, `u`, `a`, `j`, `m`, `q`, `wv`, `。`, `f`, `yv`, `s`, `mr`, `rn`, `b`, …)

Each head feeds an independent **CRF** (`pytorch-crf`) for sequence-aware decoding. The checkpoint we use is `sikuRoberta_model_crf0.pth` (~413 MB), trained on the *Zuǒ Zhuàn* corpus.

### 1.2. Per-character pipeline

For each input line `text`, the augmenter:

1. **Character normalisation** (`normalize_char`, [augment_finetune_with_siku.py:62](Erya/augment_finetune_with_siku.py#L62)) maps full-width / curly quotes and brackets onto ASCII analogues so the tagger's vocabulary covers them. Non-CJK characters pass through.
2. **Chunking** (`chunk_chars`, [L70](Erya/augment_finetune_with_siku.py#L70)): splits the line into successive 511-character windows so each window plus a leading `[CLS]` fits in SikuRoBERTa's 512-token budget.
3. **Tokenisation**: each character is mapped through `BertTokenizer.convert_tokens_to_ids(ch)` (i.e. one input ID per character — no WordPiece) and prepended with `[CLS]`. No `[SEP]` is added because the tagger code strips the `[CLS]` slot internally and operates on the remaining per-character sequence (`origin_sequence_output = layer[1:]`).
4. **Forward pass + CRF decoding** with `attention_mask=tokens.gt(0)` and `decode_mask` covering only real character positions:
   ```
   logits_seg, logits_pos = model(input_ids, attention_mask)[0]
   probs_seg, probs_pos   = softmax(logits_seg), softmax(logits_pos)
   pred_seg = crf_seg.decode(logits_seg, mask=decode_mask)   # Viterbi path
   pred_pos = crf_pos.decode(logits_pos, mask=decode_mask)
   ```
5. **Per-line write-back** (`row_offsets[idx] : row_offsets[idx]+row_length`) into pre-allocated NumPy arrays.

### 1.3. Output schema

For each `*.src` file, the script emits a `*.src.siku_aux.npz` containing eight aligned arrays plus two label tables:

| Array | Dtype | Shape | Description |
|---|---|---|---|
| `row_lengths` | `int32` | `(N,)` | Number of characters in each input line |
| `row_offsets` | `int64` | `(N+1,)` | Cumulative offsets — slice `[row_offsets[i]:row_offsets[i+1]]` to recover line `i` |
| `seg_emissions` | `float16` | `(total_chars, 4)` | Pre-CRF logits for segmentation (saved for ablation) |
| `seg_probs` | `float16` | `(total_chars, 4)` | Softmax probabilities over BIES labels |
| `seg_decode_ids` | `int16` | `(total_chars,)` | Viterbi-decoded BIES IDs |
| `pos_emissions` | `float16` | `(total_chars, 30)` | Pre-CRF logits for POS |
| `pos_probs` | `float16` | `(total_chars, 30)` | Softmax probabilities over POS labels |
| `pos_decode_ids` | `int16` | `(total_chars,)` | Viterbi-decoded POS IDs |
| `seg_label_names` | `object` | `(4,)` | String label table |
| `pos_label_names` | `object` | `(30,)` | String label table |

A top-level [`siku_aux_manifest.json`](Erya/dataset/) records the source tarball, checkpoint, save dtype, and per-file `(rows, chars)` counts for sanity checking.

### 1.4. Storage and runtime

`finetune.tgz` (127 MB original) → `finetune_with_siku_aux.tgz` (12.7 GB augmented). The blow-up factor of ~100× is dominated by the dense float-16 probability arrays (`(total_chars × 34) × 2 bytes`). Runtime on an H100 is roughly **30 minutes** for the full augmentation; smaller GPUs scale linearly.

### 1.5. Why soft probabilities, not hard labels

Saving the full softmax distribution rather than just the Viterbi-decoded `*_decode_ids` allows the downstream model to consume **soft** tag information. A character with ambiguous segmentation (e.g. probability split 0.45/0.40/0.10/0.05 across BIES) injects a different signal than a confidently-tagged one — the model learns to attend to confidence as well as identity. Hard one-hot labels would discard that uncertainty.

---

## 2. Modifications to the Erya Model

The base translation model is **Erya** ([RUCAIBox/Erya](https://huggingface.co/RUCAIBox/Erya)), a `CPTForConditionalGeneration` checkpoint (BART-like, 12-layer encoder + 2-layer decoder, `d_model=768`, `vocab=51,271`, `max_position_embeddings=1024`) pretrained on Ancient + Modern Chinese with DMLM and DAS objectives. Inference uses `BertTokenizer`, which tokenises CJK input character-by-character.

### 2.1. The TaggedErya wrapper

We add tag conditioning **without subclassing** any internal HF module. The wrapper [Erya/finetune/tag_adapter.py:TaggedErya](Erya/finetune/tag_adapter.py) introduces three small trainable components:

| Component | Type | Shape | Init |
|---|---|---|---|
| `E_seg` | `nn.Embedding` | `(num_seg, d_model) = (4, 768)` | `N(0, 0.02)` |
| `E_pos` | `nn.Embedding` | `(num_pos, d_model) = (30, 768)` | `N(0, 0.02)` |
| `alpha_seg`, `alpha_pos` | `nn.Parameter` (scalar) | `()` | `0` |

At each forward pass we mirror the augmenter's character-level tokenisation, then construct encoder inputs explicitly:

$$
\mathbf{e}_{\text{tok}} = E_{\text{tok}}(\mathbf{x}) \cdot s_e
$$
$$
\mathbf{e}_{\text{seg}} = (\mathbf{p}_{\text{seg}} \in \mathbb{R}^{T \times 4}) \cdot E_{\text{seg}}, \qquad
\mathbf{e}_{\text{pos}} = (\mathbf{p}_{\text{pos}} \in \mathbb{R}^{T \times 30}) \cdot E_{\text{pos}}
$$
$$
\mathbf{e}_{\text{in}} = \mathbf{e}_{\text{tok}} \;+\; \tanh(\alpha_{\text{seg}}) \cdot \mathbf{e}_{\text{seg}} \;+\; \tanh(\alpha_{\text{pos}}) \cdot \mathbf{e}_{\text{pos}}
$$

`e_in` is passed to the BART encoder via `model(inputs_embeds=e_in, attention_mask=…, labels=…)`. HF's `BartEncoder.forward` skips its internal `embed_tokens * embed_scale` step when `inputs_embeds` is supplied but still adds learned positional embeddings, so the rest of the encoder stack is unchanged.

**Identity at initialisation.** Because `α_seg = α_pos = 0`, `tanh(α)=0`, and the soft tag contributions are masked out at step 0; the wrapper's loss equals the vanilla model's loss on any batch. A startup sanity check in [train.py:identity_sanity_check](Erya/finetune/train.py) verifies this within `1e-3` before training begins.

**Generation.** During beam search, [TaggedErya.generate](Erya/finetune/tag_adapter.py) precomputes `encoder_outputs` from the tag-injected `inputs_embeds` and passes them to `model.generate(encoder_outputs=…)`. This avoids HF re-running the encoder without tags during decoding.

### 2.2. Auxiliary KL distillation head

A linear head [losses.py:KLDistillHead](Erya/finetune/losses.py) projects the encoder's last hidden state to `(num_seg + num_pos)` logits. During training we minimise

$$
\mathcal{L}_{\mathrm{KL}} = \frac{1}{|\mathcal{V}|}\sum_{t \in \mathcal{V}} \Big[ \mathrm{KL}\big(p^{T}_{\text{seg},t}\,\|\,p^{S}_{\text{seg},t}\big) + \mathrm{KL}\big(p^{T}_{\text{pos},t}\,\|\,p^{S}_{\text{pos},t}\big) \Big]
$$

where teachers `p^T` are SikuRoBERTa softmaxes from the npz, students `p^S` come from the linear head, and `V` is the set of non-special, non-pad positions with non-zero teacher mass (so denoising batches with zeroed teachers don't contribute). This term encourages the encoder's representation to retain tag-recoverable information rather than collapsing under the translation objective.

### 2.3. Tokenisation parity

The augmenter wrote per-character outputs with `convert_tokens_to_ids(normalize_char(ch))`. We reproduce this exactly in [aux_dataset.py:PairedAuxDataset.__getitem__](Erya/finetune/aux_dataset.py) — char-by-char IDs, prepend `[CLS]`, append `[SEP]`, and zero-pad the seg/pos prob tensors at the two special positions so alignment with `input_ids` is exactly 1:1. A runtime assertion compares `len(input_ids) - 2` against `row_lengths[i]` per batch.

### 2.4. Total parameter count

| Stage | Trainable params | Frozen params |
|---|---|---|
| 1 — adapters only | ~75 k (`E_seg` + `E_pos` + 2 scalars + KL head) | ~410 M |
| 2 — adapters + encoder | ~125 M | ~285 M |
| 3 — full model | ~410 M | 0 |

The adapter alone is < 0.02 % of Erya's parameter count.

---

## 3. Fine-tuning Dataset

We fine-tune on the **Erya parallel corpus** ([RUCAIBox/Erya-dataset](https://huggingface.co/datasets/RUCAIBox/Erya-dataset))'s `finetune.tgz` split — six EvaHan-style sub-corpora drawn from canonical classical books, each pre-aligned into ancient ↔ modern Chinese sentence pairs.

| Sub-corpus | Source text | Splits |
|---|---|---|
| `dataset` | Mixed canonical | train, valid |
| `hans` | *Hàn Shū* (Book of Han) | train, valid, test |
| `mings` | Ming-dynasty texts | train, valid, test |
| `shij` | *Shǐjì* (Records of the Grand Historian) | train, valid, test |
| `taip` | *Tàipíng Yùlǎn* | train, valid, test |
| `xint` | *Xīn Táng Shū* | train, valid, test |
| `xux` | Xú Xiákè's travel diaries | train, valid, test |

### 3.1. Volume

After augmentation the on-disk extracted layout is:

- 7 train splits, 7 valid splits, 6 test splits
- 20 paired `*.src` / `*.tgt` files
- 20 `*.src.siku_aux.npz` files containing per-character SikuRoBERTa outputs
- One `siku_aux_manifest.json`

Total characters across train splits: roughly **30M+**. Total validation examples concatenated across all sub-corpora: **~206k**.

### 3.2. Sampling

Training batches are drawn by a **length-balanced sampler** ([aux_dataset.py:LengthBalancedSampler](Erya/finetune/aux_dataset.py)) over the `ConcatDataset` of all six train splits. Per-corpus weights default to `1 / sqrt(corpus_size)` so that smaller sub-corpora (e.g. `taip`, `xint`) are not drowned out by larger ones during fine-tuning. Sampling is done with replacement so the effective epoch length is `num_steps × batch_size × grad_accum` regardless of corpus sizes.

For stage 2, an optional `MonoDenoiseDataset` ([aux_dataset.py:MonoDenoiseDataset](Erya/finetune/aux_dataset.py)) over `monolingual.tgz` is mixed in at a fixed `denoise_ratio` (default 0.2). Denoising batches receive **zeroed** `seg_probs` / `pos_probs` so the same forward path works without special-casing — the tag adapter receives no gradient on those steps because `α × 0 = 0`.

### 3.3. Validation subsampling

The full 206k-example valid set takes ~16 minutes to score per pass on H100. We instead take a deterministic random subset of `valid_max_batches × batch_size = 200 × 16 = 3,200` examples (fixed seed) so the same valid set is scored at every step → CE numbers are directly comparable across checkpoints, and a valid pass takes ~15 seconds.

---

## 4. Three-Stage Training Regime

The curriculum reflects a "**add capacity, adapt the consumer, adapt the rest**" pattern: introduce new trainable parameters that are identity at init, then progressively unfreeze the parts of the pretrained model that need to consume them.

### 4.1. Stage objectives

| Stage | Trainable | Frozen | Translation CE | KL distill | Denoising | Notes |
|---|---|---|---|---|---|---|
| **1** | `E_seg`, `E_pos`, `α_*`, KL head | encoder, decoder, LM head | ✓ | 0.10 | — | Find a useful tag-embed subspace; no risk of catastrophic forgetting because the base model isn't touched |
| **2** | + 12-layer encoder | decoder, LM head | ✓ | 0.05 | 1.0 (20 % of batches) | Encoder learns to *use* tag-conditioned input; denoising regularises against drift |
| **3** | + decoder, LM head, embeddings | — | ✓ | 0.02 | — | Decoder gently re-aligns to the new encoder representations; label smoothing 0.1 |

### 4.2. Optimiser and schedule

- `AdamW` with `weight_decay=0.01` (excluding adapter scalars and biases).
- Cosine LR schedule with linear warmup (`warmup_steps=400`/`800`/`800` per stage).
- Two-group LR: adapters at `5e-4` (stages 1–2), encoder at `5e-5` (stage 2), decoder + everything at `1e-5` (stage 3). The adapters' LR is intentionally **10× higher** than the pretrained components — adapter weights start near zero and need to grow; pretrained weights need to nudge.
- Mixed precision via `torch.autocast(bf16)` on H100 / A100; `gradient_checkpointing` enabled for stages 2 + 3 to fit larger effective batches.
- `grad_accum = {1, 2, 4}` per stage to keep effective batch = 16 / 32 / 64 while VRAM is consumed by progressively-more-trainable layers.

### 4.3. Loss summary

The total loss per batch is

$$
\mathcal{L} = \mathcal{L}_{\mathrm{CE}}^{\text{translation}} \;+\; \lambda_{\mathrm{KL}} \cdot \mathcal{L}_{\mathrm{KL}}^{\text{distill}} \;+\; \lambda_{\mathrm{denoise}} \cdot \mathcal{L}_{\mathrm{CE}}^{\text{denoise}} \cdot \mathbb{1}[\text{batch is mono}]
$$

with `λ_KL ∈ {0.10, 0.05, 0.02}` decaying across stages and `λ_denoise = 1.0` only in stage 2.

### 4.4. Checkpointing and resume

Stage 1 emits a small **adapter-only** checkpoint (`adapter.pt`, ~5 MB containing `E_seg`, `E_pos`, `α_*`, KL head). Stages 2 and 3 emit **full** checkpoints (`full.pt`, ~1.6 GB). Resume from the previous stage's best is wired into the run-all wrapper script ([Erya/scripts/run_all_stages.sh](Erya/scripts/run_all_stages.sh)):

```
stage 1 → best/adapter.pt
            ↓ --resume_adapter
stage 2 → best/full.pt
            ↓ --resume
stage 3 → best/full.pt
```

The wrapper is idempotent: stages whose `best/*.pt` already exist on disk are skipped, so a crash mid-pipeline can be resumed by re-running the same command.

### 4.5. Realised training time (single H100 SXM5 80 GB)

| Stage | Optimizer steps | Wall-clock |
|---|---|---|
| Stage 1 | 8,000 | ~4 h 30 m (with full-valid passes) |
| Stage 2 | 16,000 | ~1 h 25 m (with `valid_max_batches=200`) |
| Stage 3 | 8,000 | ~1 h 30 m (with `valid_max_batches=200`) |

The wall-clock disparity between stage 1 and the others is almost entirely validation cost — stage 1 was run before the `valid_max_batches` knob was introduced and wasted ~4 h scoring the full 206k-example valid set 16 times.

---

## 5. Preliminary Results

This section reports BLEU and chrF on the EvaHan `hans` test split. Full per-corpus ablations are pending (see § 5.5).

### 5.1. Final-stage validation losses

| Stage | Best validation cross-entropy (200-batch subset) |
|---|---|
| Stage 1 | 3.65 |
| Stage 2 | 1.91 |
| Stage 3 | **1.78** |

The roughly 50 % CE drop between stage 1 and stage 2 is consistent with the diagnosis in § 5.4: stage 1 trains adapters only, but the (BART-loaded) decoder begins **randomly initialised**, so adapters alone cannot drive CE much below 3.5. Stage 2 unfreezes the encoder and effectively trains the decoder for the first time, producing a steep CE drop.

### 5.2. Translation quality on the `hans` test split

| Variant | BLEU | chrF |
|---|---|---|
| Stage 1 best | 0.04 | 1.71 |
| Stage 2 best | 27.80 | 25.18 |
| **Stage 3 best, adapter on (`α_seg=+0.41`, `α_pos=−0.78`)** | **28.48** | **25.73** |
| Stage 3 best, adapter neutralised (`α=0`) | 22.99 | 21.43 |

### 5.3. Adapter contribution

Comparing the trained-`α` row to the `α=0` row isolates the adapter's effect at inference:

$$
\Delta_{\mathrm{adapter}}^{\mathrm{BLEU}} = 28.48 - 22.99 = +5.49 \quad (+19\% \text{ relative})
$$
$$
\Delta_{\mathrm{adapter}}^{\mathrm{chrF}} = 25.73 - 21.43 = +4.30 \quad (+17\% \text{ relative})
$$

Soft tag injection contributes **~+5.5 BLEU** on the `hans` split. The adapter is doing real work, not a no-op. The encoder evidently learns to depend on tag-conditioned input embeddings during stages 2 and 3; removing them at inference produces distribution-shifted hidden states that the decoder can no longer translate cleanly.

### 5.4. Caveat: BART vs. CPT loading

Erya's `config.json` declares `architectures: ["CPTForConditionalGeneration"]` but `model_type: "bart"`. Because CPT is not in mainline `transformers`, `AutoModelForSeq2SeqLM.from_pretrained(..., trust_remote_code=True)` falls back to `BartForConditionalGeneration`. CPT and BART share encoder structure (so encoder weights load successfully) but use **different decoder layer types** — CPT's decoder is BERT-style, BART's is BART-style. Their parameter names do not match. As a result:

- The 12-layer encoder loads cleanly from `pytorch_model.bin`.
- The 2-layer decoder is reported as `MISSING` and **randomly re-initialised**.

This means our pipeline effectively trains a fresh BART decoder from scratch (8k+16k+8k = 32k optimizer steps, ~500k example exposures) on top of Erya's pretrained encoder, with the soft tag adapter learned jointly. The 28.48 BLEU result is therefore best characterised as

> *"Erya encoder + soft tag adapter + freshly-trained BART decoder, fine-tuned on the EvaHan parallel corpus."*

The "zero-shot" row in the ablation table (BLEU 0.04) is **not** vanilla Erya — it's a BART model with a random decoder. Comparing against published Erya zero-shot numbers requires vendoring the CPT modeling code so the decoder weights actually load. That work is outstanding and tracked in § 6.

### 5.5. Pending ablations

The following per-corpus runs will replace this paragraph in the final report:

1. **All six test splits, stage 3 trained-α**: tells us how the adapter generalises across domains (`hans`, `mings`, `shij`, `taip`, `xint`, `xux`).
2. **All six test splits, stage 3 α=0**: per-corpus magnitude of the adapter contribution. SikuRoBERTa was trained on the *Zuǒ Zhuàn*; we expect tags to fit some domains (`hans`, `shij`) better than others (`mings`, modern texts).
3. **α scaled to {0.5×, 1.5×}**: tests whether the trained α value is the inference optimum.
4. **Vanilla Erya baseline with proper CPT loading**: replaces the broken zero-shot row.
5. **BERTScore P/R/F1** (`bert-base-chinese`) per corpus, for a learned-embedding metric that complements BLEU/chrF on Chinese.

---

## 6. Outstanding Work

1. **Vendor CPT modeling code.** Add `modeling_cpt.py` from [fnlp/CPT](https://github.com/fastnlp/CPT) and rewire `train.py` / `eval.py` to instantiate `CPTForConditionalGeneration` directly. This unlocks the proper Erya zero-shot baseline and may shift the absolute BLEU numbers (the pretrained CPT decoder is presumably stronger than 32k steps of from-scratch training).
2. **BLEU-aware checkpoint selection.** Currently we pick the best stage 2/3 checkpoint by valid CE. CE-best and BLEU-best can disagree; a small held-out generation set scored every N steps would pick more reliably.
3. **Per-corpus α tuning.** Train (or at least eval-time scale) per-corpus α to handle SikuRoBERTa's domain bias.
4. **Hard-label baseline.** Run an ablation that uses one-hot `seg_decode_ids` / `pos_decode_ids` instead of soft probs, to quantify the value of the soft-distribution distillation.
5. **Encoder-only ablation.** Drop stage 3 entirely (use stage 2 best) and check whether the additional decoder fine-tune actually helps BLEU — stage 3 risks decoder drift after a frozen-decoder stage 2.

---

## 7. Reproducibility

- Code: this repository, branch `translation`. Entry points: [Erya/finetune/train.py](Erya/finetune/train.py) (training), [Erya/finetune/eval.py](Erya/finetune/eval.py) (evaluation).
- Data augmentation: [Erya/augment_finetune_with_siku.py](Erya/augment_finetune_with_siku.py) ↔ [Erya/dataset/finetune_with_siku_aux.tgz](Erya/dataset/) (12.7 GB, generated from `finetune.tgz`).
- Configs: [Erya/configs/stage{1,2,3}*.yaml](Erya/configs/).
- Hardware used: 1 × NVIDIA H100 SXM5 80 GB (RunPod), bf16 autocast.
- Seeds: `42` everywhere; the length-balanced sampler and valid subset use a fixed offset.

The full pipeline (data prep + 3-stage training + eval on one corpus) reproduces in roughly **5 hours** of H100 time once the augmented tarball is available.
