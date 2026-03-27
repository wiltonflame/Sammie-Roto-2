# Changes in This Fork vs Original Sammie-Roto 2

This document describes all modifications and new features in this fork compared to the original [Zarxrax/Sammie-Roto-2](https://github.com/Zarxrax/Sammie-Roto-2) codebase. It is intended to support review and potential merge into the upstream repository.

---

## Summary

| Area | Original | This Fork |
|------|----------|-----------|
| **Matting engines** | MatAnyone, MatAnyone 2 | + **VideoMaMa** (diffusion-based, optional) |
| **Tabs** | Segmentation, Matting, Object Removal | + **Corridor Key** (AI green screen keying) |
| **View modes** | 6 modes | + CK-Alpha, CK-FG, CK-Comp |
| **Export output types** | Segmentation/Matting/Removal | + **CK-Alpha**, **CK-FG**, **CK-Comp** (16-bit PNG/EXR) |
| **Segmentation / tracking** | Same as upstream | See **§4** — multi-point masks, temporal stabilize option, session resume, UX notes |
| **CorridorKey storage** | N/A (fork-only) | Writable user data path on macOS/.app (see **§2.1**); MPS + CUDA |
| **Dependencies** | requirements.txt (no antlr4 pin) | + `antlr4-python3-runtime==4.9.3`, `einops`, `huggingface_hub`, `timm`, `transformers` |
| **Launch (Linux)** | `run_sammie.command` basic | + `QT_QPA_PLATFORM=xcb`, `QT_X11_NO_MITSHM=1`, `CORRIDORKEY_OPT_MODE=lowvram` |

New models (VideoMaMa, CorridorKey) are **downloaded on first use**, so installation size and time are unchanged until the user runs those features.

---

## 1. VideoMaMa (Mask-Guided Video Matting)

### What It Is

- **VideoMaMa** is a diffusion-based video matting model (Stable Video Diffusion fine-tuned for matting).  
- **Source:** [cvlab-kaist/VideoMaMa](https://github.com/cvlab-kaist/VideoMaMa) (CVPR 2026).  
- It produces more temporally consistent mattes than frame-by-frame MatAnyone, at the cost of higher VRAM and longer processing.  
- It requires a **segmentation mask on all frames** as guidance (run Segmentation → Propagate first).

### How It Works in the App

- In the **Matting** tab, the user can choose the engine: **MatAnyone**, **MatAnyone 2**, or **VideoMaMa**.
- When VideoMaMa is selected and the user clicks **Run Matting**, the app:
  1. Ensures models are downloaded (SVD base + VideoMaMa UNet, ~19 GB with `allow_patterns` to avoid redundant files).
  2. Runs the VideoMaMa pipeline (batch of 16 frames, overlap/blending for longer videos).
  3. Writes mattes under `temp/matting/` per frame/object; the viewer shows **Matting-Matte** as today.

### Main Files for VideoMaMa

| File | Role |
|------|------|
| **`videomama/`** (new directory) | VideoMaMa integration |
| `videomama/__init__.py` | Package init |
| `videomama/videomama_manager.py` | Manager: download (HuggingFace), run pipeline, progress UI, VRAM cleanup |
| `videomama/pipeline_svd_mask.py` | SVD + mask pipeline (diffusers), batch inference |

### Files Modified to Support VideoMaMa

| File | Changes |
|------|---------|
| `sammie_main.py` | MattingTab: engine combo (MatAnyone / MatAnyone 2 / VideoMaMa), `run_matting()` branches to VideoMaMaManager, VRAM cleanup before load, warning for low VRAM |
| `sammie/settings_manager.py` | `default_matting_engine`, `default_videomama_overlap`, `default_videomama_resolution`, `default_videomama_vae_tiling`; SessionSettings + persistence for VideoMaMa |
| `sammie/sammie.py` | View/export handling for matting output (unchanged for existing MatAnyone; VideoMaMa writes same matting paths) |
| `requirements.txt` | `diffusers`, `accelerate`, `huggingface_hub`, etc. (already in original); no extra package beyond existing stack |

---

## 2. Corridor Key (New Tab and Pipeline)

### What It Is

- **CorridorKey** is an AI green screen keyer: it refines a coarse matte (from Segmentation or Matting), outputs **refined alpha**, **despilled foreground**, and **composite**.  
- **Source:** [EZ-CorridorKey](https://github.com/edenaion/EZ-CorridorKey) (GUI by Ed Zisk; upstream [nikopueringer/CorridorKey](https://github.com/nikopueringer/CorridorKey)).  
- Inference logic and low-VRAM behavior (tiling, quality presets) follow the EZ-CorridorKey approach, integrated as a dedicated tab and export types.

### Tab: Corridor Key

- New sidebar tab **Corridor Key** with:
  - **Run CorridorKey** / **Clear CorridorKey**
  - **Mask Source:** Segmentation | Matting (which mask to refine).
  - **Quality:** Low (1024) | Medium (1536) | High (2048) — internal resolution; auto by VRAM if desired.
  - **Refiner Scale:** 0–2 (edge refinement strength).
  - **Despill:** 0–1 (green spill removal on FG).
  - **Auto Despeckle** + **Despeckle Size** (pixel threshold).
  - **Tiling:** optional tile-based refiner for low VRAM.

### View Modes and Export

- New view modes: **CK-Alpha**, **CK-FG**, **CK-Comp** (viewer reads from `corridorkey/` outputs).
- **File Export → Output Type:** added **CK-Alpha**, **CK-FG**, **CK-Comp** for all relevant formats (video and image sequences, including EXR and 16-bit PNG where applicable).

### Outputs and Quality

- Alpha, FG, and Comp are saved as **16-bit PNG** in a `corridorkey/` subfolder; an 8-bit alpha copy is still written for the existing matting viewer path if needed.
- EXR sequence export supports CK-Alpha, CK-FG, CK-Comp (float/RGB layers as appropriate).

### 2.1 macOS / packaging note — checkpoints path (not SAM2)

The **original** Sammie SAM2 model flow (e.g. `checkpoints/` next to the project) already works for typical macOS dev installs. The **`[Errno 13] Permission denied: 'checkpoints/CorridorKey'`** issue appeared because **CorridorKey** (this fork’s addition) initially used a **cwd-relative** path `checkpoints/CorridorKey`. That fails when:

- The app runs inside a **read-only** `.app` bundle, or  
- The **current working directory** is not the project root / not writable.

**Fix in this fork:**

- New module **`corridorkey/paths.py`** with `get_corridorkey_checkpoint_dir()`:
  - Prefer **`checkpoints/CorridorKey` under the repository root** only if it can be **created and is writable** (normal git clone on a writable disk).
  - Otherwise use a **user-writable** directory:
    - **macOS:** `~/Library/Application Support/Sammie-Roto/checkpoints/CorridorKey`
    - **Windows:** `%LOCALAPPDATA%\Sammie-Roto\checkpoints\CorridorKey`
    - **Linux:** `$XDG_DATA_HOME/Sammie-Roto/...` or `~/.local/share/Sammie-Roto/...`
- `os.makedirs` wrapped with a clear error if the directory still cannot be created.

**Apple Silicon:** CorridorKey **load/run** accept **`mps`** as well as **`cuda`** (CPU-only remains unsupported). Auto quality on MPS uses a sensible default internal resolution (unified memory). GPU cache between frames uses `torch.mps.empty_cache()` when MPS is available, in addition to CUDA paths.

### Main Files for Corridor Key

| File | Role |
|------|------|
| **`corridorkey/`** (new directory) | CorridorKey integration |
| `corridorkey/__init__.py` | Package init |
| `corridorkey/corridorkey_manager.py` | Manager: HuggingFace download, load/unload engine, run per-frame, save 16-bit PNG + 8-bit alpha copy; **MPS + CUDA**; safe GPU cache release |
| `corridorkey/paths.py` | **Checkpoint directory:** writable location (avoids `Permission denied` on macOS .app / read-only CWD); see **§2.1** |
| `corridorkey/inference_engine.py` | Inference: load checkpoint, refiner (full-frame or tiled), despill, despeckle; `torch.compile` disabled on Linux+Qt to avoid segfaults; MPS-aware cache clear on compile fallback |
| `corridorkey/core/model_transformer.py` | Model (Hiera) and refiner architecture |
| `corridorkey/core/color_utils.py` | Color/despill utilities |

### Files Modified to Support Corridor Key

| File | Changes |
|------|---------|
| `sammie_main.py` | New **CorridorKeyTab** and Sidebar tab "Corridor Key"; `run_corridorkey()` / `clear_corridorkey()`; view combo + CK-Alpha/CK-FG/CK-Comp; tab change switches view to CK-Alpha when Corridor Key tab selected; Quality/Tiling from settings; VRAM cleanup before load; `clear_corridorkey` resets view to Segmentation-Edit if current view was CK-*; **CorridorKey allows `mps` (Apple Silicon), not CUDA-only**; `torch.mps.empty_cache()` where relevant |
| `sammie/settings_manager.py` | `default_corridorkey_*` (mask_source, refiner_scale, despill, despeckle, despeckle_size, quality, tiling); SessionSettings + persistence |
| `sammie/sammie.py` | `_load_ck_output()`, `_handle_ck_alpha_view`, `_handle_ck_fg_view`, `_handle_ck_comp_view` (16-bit PNG read, normalize for display/export); object_id_filter for single-object export; DeviceManager `clear_cache()` with `gc.collect()` for VRAM |
| `sammie/export_formats.py` | `get_available_output_types()` extended with **CK-Alpha**, **CK-FG**, **CK-Comp** for all video and sequence formats (ProRes, FFV1, H.264, H.265, VP9, PNG seq, EXR seq) |
| `sammie/export_workers.py` | Export for CK-*: 16-bit PNG handling, EXR layers for CK-FG/CK-Comp; `has_alpha` limited to Segmentation-Alpha/Matting-Alpha so CK-Alpha is not treated as RGBA alpha channel |

---

## 3. Other Modifications (Stability, UX, Install)

### Requirements and Install

| File | Change |
|------|--------|
| `requirements.txt` | Pinned `antlr4-python3-runtime==4.9.3` for Hydra/SAM2 compatibility (avoids ATN deserialization segfault); added `einops`, `huggingface_hub`, `timm`, `transformers` as needed by new models. |

### Launch Script (Linux)

| File | Change |
|------|--------|
| `run_sammie.command` | On Linux: `QT_QPA_PLATFORM=xcb`, `QT_X11_NO_MITSHM=1` to reduce Qt/OpenGL segfaults; `CORRIDORKEY_OPT_MODE=lowvram` so CorridorKey does not enable torch.compile (which can segfault with Qt on Linux). |

### UI / Behavior

- **Sidebar:** minimum width increased; tab content wrapped in `QScrollArea` so Corridor Key (and other tabs) are scrollable when the window is small; `on_tab_changed` unwraps scroll area to identify active tab.
- **Main window:** default size and main splitter proportions adjusted so the Corridor Key tab is not clipped; optional centering on screen; minimum window size set.
- **VRAM:** `gc.collect()` before `torch.cuda.empty_cache()` in DeviceManager, CorridorKey and VideoMaMa load/unload paths to reduce fragmentation and OOM on heavy use.

### Optional / Not in Repo

- `install_dependencies_flame.sh`: custom installer for Autodesk Flame (Python path, CUDA choice). Can stay local or be contributed separately.
- `clean_for_git.sh`, `GIT_SUBMIT.md`: ignored via `.gitignore` (local workflow only).
- `.gitignore`: `checkpoints/` fully ignored; `clean_for_git.sh`, `GIT_SUBMIT.md` ignored.

---

## 4. Segmentation, tracking, and UX (fork refinements)

These changes address behaviours reported against upstream (e.g. masks degrading with many points on one object, temporal jitter after propagate) and clarify SAM2 usage. They do **not** replace the upstream SAM2 download layout on macOS.

### 4.1 Interactive segmentation (multi-point stability)

Aligned with discussion around [upstream issue #47](https://github.com/Zarxrax/Sammie-Roto-2/issues/47) (holey / unstable masks when adding many points):

| File | Change |
|------|--------|
| `sam2/sam2_video_predictor.py` | Re-enabled **`fill_holes_in_mask_scores`** when `fill_hole_area` is positive (it had been commented out while `build_sam.py` still set `fill_hole_area=8`, so small interior holes were not filled like in `sam2_video_predictor_legacy`). |
| `sammie/sammie.py` | **`segment_image`:** removed **`reset_state`** before every `add_new_points_or_box`. Resetting wiped prior SAM outputs, so each click re-ran all prompts from scratch without mask refinement and without single-click multimask behaviour. **`replay_points`** and **`clear_tracking`** still reset when a full rebuild is required. |
| `sammie_main.py` | **Resume session:** if saved points exist, call **`replay_points`** after `initialize_predictor` so the first new click is not applied to an empty predictor state. |

### 4.2 Propagated masks — optional temporal stabilization

| File | Change |
|------|--------|
| `sammie/sammie.py` | **`_stabilize_propagated_mask_logits`:** during **`track_objects`** only, optional EMA over time (per object), light Gaussian blur on probabilities, small morphological close — reduces edge flicker between frames. |
| `sammie/settings_manager.py` | Session field **`tracking_stabilize_masks`** (default `True`). |
| `sammie_main.py` | Segmentation tab → **Tracking:** checkbox **“Stabilize tracked masks (less flicker between frames)”**; persists session settings. |

Interactive click segmentation and replay are unchanged by this stabilize path (only **Track Objects** output PNGs).

### 4.3 Object ID tooltip (multi-region behaviour)

SAM 2 Video tracks **one mask and one memory stream per object ID**. Several **disconnected** regions under the **same** ID force a single mask to satisfy all points, which increases ambiguity and temporal instability versus using **one ID per separate instance**. The **Object** spinbox tooltip in the Segmentation tab was expanded to document this recommended workflow.

---

## 5. File Tree Summary

```
Sammie-Roto-2/
├── CHANGES_FORK.md         # This document (fork vs upstream)
├── sammie_main.py          # VideoMaMa + Corridor Key UI; CK mps/cuda; tracking stabilize checkbox; session replay
├── run_sammie.command      # + Linux Qt env vars and CORRIDORKEY_OPT_MODE
├── requirements.txt        # + antlr4-python3-runtime, einops, huggingface_hub, timm, transformers
├── README.md               # Updated: VideoMaMa + CorridorKey, acknowledgements, links
├── sam2/
│   └── sam2_video_predictor.py  # fill_holes_in_mask_scores re-enabled (see §4.1)
├── videomama/              # NEW
│   ├── __init__.py
│   ├── videomama_manager.py
│   └── pipeline_svd_mask.py
├── corridorkey/            # NEW
│   ├── __init__.py
│   ├── paths.py            # Writable checkpoint dir (macOS .app / CWD); §2.1
│   ├── corridorkey_manager.py
│   ├── inference_engine.py
│   └── core/
│       ├── __init__.py
│       ├── color_utils.py
│       └── model_transformer.py
└── sammie/
    ├── sammie.py           # CK views, 16-bit load, DeviceManager gc; segment_image / track stabilize; §4
    ├── settings_manager.py # VideoMaMa, CorridorKey, tracking_stabilize_masks
    ├── export_formats.py   # + CK-Alpha, CK-FG, CK-Comp in output types
    └── export_workers.py   # + 16-bit PNG/EXR for CK-*; has_alpha fix
```

---

## 6. References

- **VideoMaMa:** [https://github.com/cvlab-kaist/VideoMaMa](https://github.com/cvlab-kaist/VideoMaMa)  
- **EZ-CorridorKey:** [https://github.com/edenaion/EZ-CorridorKey](https://github.com/edenaion/EZ-CorridorKey)  
- **CorridorKey (upstream):** [https://github.com/nikopueringer/CorridorKey](https://github.com/nikopueringer/CorridorKey)  

---

## Document history

| Date | Notes |
|------|--------|
| *(initial)* | VideoMaMa, Corridor Key, export/view integration, Linux launch vars, requirements pins. |
| **2026-03-27** | §2.1 macOS CorridorKey path + MPS; §4 segmentation/tracking/resume/tooltip; file tree + summary table updated. Clarified SAM2 downloader unchanged for macOS — CorridorKey cwd-relative checkpoints caused the permission error. |

---

This document can be shared with the upstream maintainer for review. If a Pull Request is not possible, it can be sent as a summary of changes for manual consideration.
