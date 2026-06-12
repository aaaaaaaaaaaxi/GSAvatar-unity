# GSAvatar (WebAvatar Unity Package)

A Unity UPM package that loads [**WebAvatar**](https://gapszju.github.io/webavatar) (CVPR 2026) checkpoints — NPZ avatar models plus JSON pose sequences exported by the [Python training code](https://github.com/1231234zhan/webavatar) — and renders pose-animated 3D Gaussian Splatting (3DGS) avatars in real time on top of [`wu.yize.gsplat`](https://github.com/wuyize25/gsplat-unity).

> Companion viewer to the [webavatar-rust](https://github.com/aaaaaaaaaaaxi/webavatar-rust) WebGPU viewer. Same `.npz` / `.json` assets, different runtime.

## Paper

> **High-Fidelity Mobile Avatars with Pruned Local Blendshapes — WebGPU Viewer**
> Youyi Zhan, He Wang, Tianjia Shao, Kun Zhou · CVPR 2026
> [Project page](https://gapszju.github.io/webavatar) · [arXiv](https://arxiv.org/abs/2605.01854)

## Features

- **Drag-and-drop assets** — drop a `.npz` avatar model and a `.pose.json` pose sequence into `Assets/` and they're auto-imported as `WebAvatarAsset` / `WebAvatarPoseAsset`.
- **Pose-driven 3DGS rendering** — per-frame compute pipeline performs LBS skinning, MLP feature generation, attribute blend shapes and barycentric transform; output is rasterised by the gsplat-unity renderer.
- **Edit-mode preview** — scrub the `PreviewFrame` slider to inspect a single frame in the Scene view.
- **Runtime playback** — `Play / Pause`, FPS control, and Loop / Clamp / Once end-of-sequence modes.
- **No build step** — pure UPM package, no DLLs, no native plugins.

## Install

### 1. Install the base gsplat renderer

Follow [`wuyize25/gsplat-unity`](https://github.com/wuyize25/gsplat-unity) README to add the base renderer package.

### 2. Install this package

Add as a local UPM package pointing to this folder:

```json
// Packages/manifest.json
{
  "dependencies": {
    "org.webavatar.runtime": "file:../path/to/GSAvatar-unity",
    "wu.yize.gsplat": "1.1.2"
  }
}
```

### 3. Drop your assets

Copy your `.npz` and `.pose.json` files anywhere under `Assets/` (do **not** put them under `Resources/`).

## Asset format

| File | Extension | Produced asset | Source |
|---|---|---|---|
| Avatar model | `.npz` | `WebAvatarAsset` | Python training code → export NPZ |
| Pose sequence | `.pose.json` | `WebAvatarPoseAsset` | Rust/WebGPU viewer pipeline |

> Pose JSON files must use the two-segment extension `.pose.json` because Unity's native `TextScriptImporter` already owns the bare `.json` extension and would otherwise reject our `ScriptedImporter` registration.

## Usage

### Scene setup

1. Create an empty GameObject in your scene.
2. Add the **`WebAvatar/Web Avatar Renderer`** component (this auto-adds `WebAvatarComponent`).
3. Assign your imported `WebAvatarAsset` (and optional `WebAvatarPoseAsset`) in the Inspector.
4. Press Play.

### Inspector fields

| Field | Description |
|---|---|
| `Asset` | Imported `.npz` avatar model. |
| `PoseAsset` | Optional `.pose.json` pose sequence. Leave empty to render the canonical (big) pose. |
| `Fps` | Playback rate of the pose sequence. |
| `PlayMode` | End-of-sequence behaviour: `Loop` / `Clamp` / `Once`. |
| `Play` | Disable to freeze on the current pose while still rendering. |
| `SHDegree` | SH band for the gsplat shader. The shipped NPZ only contains band 0 — leave at `0`. |
| `GammaToLinear` | Convert gamma-space colors to linear before shading. |
| `PreviewFrame` (sibling component) | Frame index shown in the Scene-view gizmo when not in Play mode. |

### Programmatic access

```csharp
using WebAvatar;
using WebAvatar.Pose;

var avatar  = GetComponent<WebAvatarComponent>();
var frame   = avatar.GetPose(42);        // PoseFrame? — null if out of range
int poses   = avatar.PoseCount;          // 0 if PoseAsset is unset
bool loaded = avatar.HasAsset;
```

## Architecture

```
WebAvatarAsset (.npz)  ─┐
                        ├─► WebAvatarComponent ─► WebAvatarRenderer ─► wu.yize.gsplat
WebAvatarPoseAsset      ─┘             │
                                       ▼
                  WebAvatarGpuResources (ComputeBuffer pool)
                                       │
                                       ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │  Per-frame compute pipeline                                          │
   │                                                                      │
   │  1. LbsSkinning.compute        — skin canonical splats by G joints   │
   │  2. MlpForward.compute         — per-part feature generator         │
   │  3. AttributeBasis.compute    — load per-splat basis coefficients   │
   │  4. AttributeDxyz.compute      — decode dxyz (position offsets)      │
   │  5. AttributeTransform.compute — apply rotation/scale/color deltas   │
   └─────────────────────────────────────────────────────────────────────┘
```

- **`Runtime/Avatar/`** — `ScriptableObject` assets (`WebAvatarAsset`, `WebAvatarPoseAsset`), the renderer MonoBehaviour, GPU resource pool and CPU SMPL-X skinning math.
- **`Runtime/Npz/`** — minimal NPZ reader (NumPy `.npy` format).
- **`Runtime/Pose/`** — `PoseFrame` struct and `PosePlayer` sequencer.
- **`Runtime/Resources/`** — five compute shaders implementing the per-frame pipeline.
- **`Editor/`** — `ScriptedImporter` for `.npz` and `.pose.json`, custom inspector and a debug menu.

## Limitations

- Only SH band 0 is shipped in the NPZ, so `SHDegree` must stay at `0`.
- The pose JSON must be renamed to `.pose.json` (two-segment extension) to bypass Unity's native JSON importer.
- Float16 source data is widened to float32 on import; this roughly doubles GPU memory for the canonical buffers.

## License

See `LICENSE` in the [webavatar-rust](https://github.com/aaaaaaaaaaaxi/webavatar-rust) parent repo.

## Citation

```bibtex
@inproceedings{zhan2026webavatar,
  title  = {High-Fidelity Mobile Avatars with Pruned Local Blendshapes},
  author = {Zhan, Youyi and Wang, He and Shao, Tianjia and Zhou, Kun},
  booktitle = {CVPR},
  year   = {2026}
}
```