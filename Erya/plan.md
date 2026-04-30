# Erya + Soft POS/Seg Tag Adapters — Fine-tuning Plan

## Context

Erya ([Erya/pytorch_model.bin](Erya/pytorch_model.bin)) is a `CPTForConditionalGeneration` model (BART-like, 12-enc/2-dec, d_model=768, BertTokenizer, vocab=51,271, max_pos=1024) pretrained for Ancient → Modern Chinese translation. We have already produced `finetune_with_siku_aux.tgz` (12 GB) where every `.src` line has a per-character-aligned `.src.siku_aux.npz` carrying SikuRoBERTa segmentation/POS soft outputs.

The goal is to inject these soft tag distributions into the encoder via small trainable adapters, fine-tune in three stages, and produce a checkpoint that beats vanilla Erya on the EvaHan-style splits inside the six sub-corpora (`dataset/`, `taip/`, `mings/`, `xux/`, `hans/`, `xint/`, `shij/`).

User-confirmed scope decisions:
- Denoising: include on `monolingual.tgz` with **zero tag distributions** (no Siku run on mono).
- KL distillation: **on by default** in every stage.
- Inference: support **both** bundled SikuRoBERTa annotation and pre-computed `.npz`.
- Multi-corpus training: **length-balanced sampling** across the six sub-corpora.

## Architectural Approach

### Tag injection via `inputs_embeds`

Avoid subclassing `BartEncoder`. Compute encoder inputs ourselves and pass via the `inputs_embeds=` argument. HF `BartEncoder.forward` (`modeling_bart.py` ~L1085–1154) skips the internal `embed_tokens * embed_scale` when `inputs_embeds` is provided but still adds `embed_positions` after — perfect for our use.

```
tok       = encoder.embed_tokens(input_ids) * embed_scale     # embed_scale=1.0 (config.scale_embedding=false)
soft_seg  = seg_probs.float() @ E_seg.weight                  # (B, T, d_model)
soft_pos  = pos_probs.float() @ E_pos.weight
inputs_embeds = tok + tanh(alpha_seg) * soft_seg + tanh(alpha_pos) * soft_pos
out = model(inputs_embeds=inputs_embeds, attention_mask=mask, labels=labels)
```

Init `E_seg`, `E_pos` with `N(0, 0.02)` (matches BART `init_std`) and `alpha_seg = alpha_pos = 0`. At step 0 the wrapped model is bit-identical to vanilla Erya — verified by a sanity-check test in `train.py`.

### Tokenization (load-bearing — must mirror the augmenter exactly)

The augmenter ([augment_finetune_with_siku.py:215-223](Erya/augment_finetune_with_siku.py#L215-L223)) builds inputs as `[CLS] + [convert_tokens_to_ids(normalize_char(ch)) for ch in line]` with no `[SEP]`, and saves per-character outputs (model.py strips `[CLS]` internally). For Erya translation we tokenize the same way and append `[SEP]` (id 102) ourselves. The seg/pos probability tensors are wrapped with a leading and trailing zero-vector for `[CLS]`/`[SEP]`, giving exact 1:1 alignment with `input_ids`.

A `normalize_char` import + dataset assertion (`tokenizer.convert_tokens_to_ids(normalize_char(ch))` matches stored id at every character position) catches drift.

### Why probs, not emissions or hard ids

`seg_probs` / `pos_probs` are softmaxed CRF emissions. Bounded, additive over BIES marginals, and match the "soft" semantics from the spec. Code path supports `*_emissions` (logits with temperature) and one-hot `*_decode_ids` behind a flag for ablation, but defaults to probs.

## File Structure (under `Erya/`)

```
Erya/
  finetune/                      # NEW package, all code lives here
    __init__.py
    tag_adapter.py               # TaggedErya wrapper: E_seg, E_pos, alphas
    aux_dataset.py               # PairedAuxDataset, MonoDenoiseDataset, Collator, length-balanced sampler
    siku_inference.py            # SingleSentenceAnnotator (reuses SikuAnnotator plumbing)
    losses.py                    # KL distillation head + loss helpers
    denoise.py                   # BART-style span masking for monolingual.tgz
    train.py                     # 3-stage training entry point (--stage {1,2,3})
    eval.py                      # generate + sacrebleu / chrF, supports bundled or pre-computed tags
    utils.py                     # param groups, checkpoint I/O, label-table sanity check
  configs/
    stage1_adapters.yaml         # adapters only, lr 5e-4, 3 epochs
    stage2_encoder.yaml          # encoder + adapters, lr 5e-5 / 5e-4, 2 epochs
    stage3_full.yaml             # full model, lr 1e-5, 1 epoch
  scripts/
    extract_finetune_aux.sh      # one-off untar to dataset/finetune_aux/
    run_stage{1,2,3}.sbatch
  checkpoints/
    stage{1,2,3}/                # written at runtime
```

Reference (do not modify): [augment_finetune_with_siku.py](Erya/augment_finetune_with_siku.py) — canonical tokenization and alignment.

## Data Loader

[Erya/finetune/aux_dataset.py](Erya/finetune/aux_dataset.py)

`PairedAuxDataset(src, tgt, npz)` per split per corpus:
- On init: `np.load(npz_path, mmap_mode='r', allow_pickle=True)` (memmap large float16 arrays). Eagerly load `row_offsets`, `row_lengths`, `seg_label_names`, `pos_label_names`. Build/cache a line-byte-offset index for the `.src` and `.tgt` (`<file>.idx.npy`) on first access.
- `__getitem__(i)`: read line *i* from `.src` and `.tgt`. Build `input_ids` char-by-char via `convert_tokens_to_ids(normalize_char(ch))`, prepend `[CLS]=101`, append `[SEP]=102`. Slice `seg_probs[row_offsets[i]:row_offsets[i+1]]` (and pos), cast to float32, pad with a zero row at front and back to align with specials. Truncate to 1022 chars + 2 specials = 1024 (Erya `max_position_embeddings`); log truncations. Tokenize target with regular `BertTokenizer.__call__` (no soft tags on target).
- Per-batch collator pads `input_ids`/`attention_mask` with 0, pads `seg_probs`/`pos_probs` to `(B, T_max, num_*_labels)` with zeros, builds `labels` with `-100` on pads.

`MonoDenoiseDataset(monolingual_root)`: BART-style span masking (Poisson λ=3, 30% mask rate). Returns `seg_probs`/`pos_probs` as **all-zero** tensors of the right shape so the same forward path works with no special-casing.

`LengthBalancedSampler` over a `ConcatDataset` of the six parallel corpora: weights = `1 / sqrt(corpus_lines)` so smaller corpora are upweighted. Denoising data interleaved at a fixed ratio (`--denoise_ratio`, default 0.2 of steps).

## Training Script

[Erya/finetune/train.py](Erya/finetune/train.py) — a single entry point, stage selected by `--stage` and YAML.

1. Load Erya: `CPTForConditionalGeneration.from_pretrained("Erya/", trust_remote_code=True)`. Wrap with `TaggedErya(model, num_seg, num_pos)` from `tag_adapter.py`.
2. Sanity check: with `alpha=0`, forward on a fixed batch matches vanilla `model(input_ids=...)` to within fp32 epsilon. Abort if not.
3. Build param groups (`utils.build_param_groups`):
   - **Stage 1**: only `E_seg`, `E_pos`, `alpha_*`, KL head. All HF params `requires_grad=False`.
   - **Stage 2**: above + `model.encoder.*`. Decoder still frozen. Two LRs: adapters 5e-4, encoder 5e-5.
   - **Stage 3**: all params at 1e-5, warmup 5–10%.
4. Optimizer: `AdamW`, cosine schedule, gradient checkpointing on encoder for stages 2/3, bf16 autocast where supported (fp16 fallback).
5. Losses (combined every step):
   - **Translation CE**: from `model(...labels=...)`.
   - **KL distillation** (default ON): linear head on encoder last hidden state → `(num_seg + num_pos)` logits; `KL(teacher_probs ‖ softmax(student_logits))` at non-pad CJK positions only. Weight `--kl_weight=0.1` (decay across stages: 0.1 → 0.05 → 0.02).
   - **Denoising CE**: same as translation CE but on `MonoDenoiseDataset` batches; weighted by `--denoise_weight=1.0`.
6. Checkpointing:
   - Stage 1 emits `adapter.pt` (small: `E_seg`, `E_pos`, two scalars, KL head) + `meta.json` recording the seg/pos label tables for sanity check at load.
   - Stage 2/3 also save full `model.safetensors` plus `adapter.pt`.
   - `load_for_stage(N+1)`: re-init wrapper, then load adapter, then load encoder/decoder weights.
7. Logging: per-step CE / KL / denoise / `alpha_*`; valid CE every N steps; best-by-valid checkpoint per stage.

## Evaluation / Inference

[Erya/finetune/siku_inference.py](Erya/finetune/siku_inference.py): `SingleSentenceAnnotator` reuses the `SikuAnnotator` machinery from [augment_finetune_with_siku.py:96-116](Erya/augment_finetune_with_siku.py#L96-L116) but accepts a Python `str`, runs `chunk_chars`, batches chunks, returns `(seg_probs, pos_probs)` arrays of shape `(len(text), num_*)`. Loads SikuRoBERTa once and persists across calls.

[Erya/finetune/eval.py](Erya/finetune/eval.py):
- Two input modes: bundled (calls `SingleSentenceAnnotator`) or pre-computed (`--aux_npz path`). Default bundled.
- Per sentence: build `inputs_embeds` exactly as in training, then `model.generate(inputs_embeds=..., attention_mask=..., max_new_tokens=256, num_beams=4)`.
- Fallback if `generate` rejects `inputs_embeds`: precompute `encoder_outputs = encoder(inputs_embeds=...)` and pass `encoder_outputs=` to `generate`. (HF supports this for encoder-decoder.)
- Metrics: sacreBLEU (zh tokenization) + chrF, per sub-corpus and combined.

## Critical Files to Create/Modify

- `Erya/finetune/tag_adapter.py` — wrapper module with `E_seg`, `E_pos`, `alpha_*`
- `Erya/finetune/aux_dataset.py` — `PairedAuxDataset`, `MonoDenoiseDataset`, collator, length-balanced sampler
- `Erya/finetune/siku_inference.py` — single-sentence Siku annotator
- `Erya/finetune/losses.py` — KL head + helpers
- `Erya/finetune/denoise.py` — BART span masking
- `Erya/finetune/train.py` — 3-stage entry point
- `Erya/finetune/eval.py` — generation + BLEU/chrF
- `Erya/finetune/utils.py` — param groups, checkpoint I/O
- `Erya/configs/stage{1,2,3}.yaml`
- `Erya/scripts/run_stage{1,2,3}.sbatch`

Do NOT modify [Erya/augment_finetune_with_siku.py](Erya/augment_finetune_with_siku.py) — it is the canonical reference for tokenization and per-character alignment.

## Risks / Open Issues

1. **Tokenization replication is load-bearing.** Any drift from the augmenter's `normalize_char` + per-char id lookup silently misaligns soft tags with the encoder character stream. Mitigation: an in-loader assertion that for every example, recomputed ids match `len(input_ids) - 2 == row_lengths[i]`.
2. **`generate(inputs_embeds=...)` for `CPTForConditionalGeneration`.** Verified for HF BART; CPT inherits but uses `trust_remote_code`. Fallback path (precomputed `encoder_outputs=`) is implemented from day one.
3. **>1024-char inputs.** Augmenter writes aux for the full line, but Erya max-pos is 1024. Truncate to 1022 chars + 2 specials in the loader; for inference, chunk-translate-and-concatenate.
4. **VRAM at inference.** Erya (~410M) + SikuRoBERTa (~400M) + KV cache. Doc minimum 16 GB; bf16 both models if tighter.
5. **Denoising on mono with zero tags.** Tag adapter receives no gradient on those steps (`alpha_*` × 0 = 0). That is the desired behavior: denoising regularizes the encoder and decoder, not the adapters.
6. **`[SEP]` vs `[EOS]` discrepancy.** `tokenizer_config.json` lists `eos=[EOS]` but `config.json` sets `eos_token_id=102=[SEP]`. We trust `config.json`: append id 102 and use it as the eos for `generate`.

## Verification (end-to-end)

After Stage 1 completes:
1. **Identity sanity** (run as part of `train.py --dry_run`): with `alpha_*` set to 0 and `E_*` zeroed, `loss_tagged ≈ loss_vanilla` to fp32 epsilon on a held-out batch.
2. **Alignment sanity**: assertion in dataloader that `len(input_ids) - 2 == row_lengths[i]` and `convert_tokens_to_ids(normalize_char(ch))` matches at every position.
3. **Adapter does something**: after Stage 1, valid CE on `dataset/valid` should drop modestly vs vanilla Erya (`tools/eval.py --no_adapter` baseline). If not, suspect a sign error or zero-init issue with `alpha_*`.

After Stage 2:
4. **BLEU on each sub-corpus test split** vs vanilla Erya. Target: ≥+1 BLEU on at least four of the seven (`dataset/` lacks test).
5. **Inference path equivalence**: `eval.py --bundled` and `eval.py --aux_npz <pre-computed>` produce identical translations on the same sentences (modulo Siku determinism).

After Stage 3 (optional):
6. Re-run BLEU; expect small additional gains on harder corpora; watch for regression on smaller corpora (overfitting risk at full-model LR).

Run order:
```
bash Erya/scripts/extract_finetune_aux.sh
python -m Erya.finetune.train --stage 1 --config Erya/configs/stage1_adapters.yaml
python -m Erya.finetune.train --stage 2 --config Erya/configs/stage2_encoder.yaml --resume_adapter checkpoints/stage1/best/adapter.pt
python -m Erya.finetune.train --stage 3 --config Erya/configs/stage3_full.yaml --resume checkpoints/stage2/best/
python -m Erya.finetune.eval --ckpt checkpoints/stage3/best --bundled
```
