# GSAvatar（WebAvatar Unity 包）

一个 Unity UPM 包，用于加载 [**WebAvatar**](https://gapszju.github.io/webavatar)（CVPR 2026）的检查点——由 [Python 训练代码](https://github.com/1231234zhan/webavatar) 导出的 NPZ 化身模型和 JSON 姿态序列——并在 [`wu.yize.gsplat`](https://github.com/wuyize25/gsplat-unity) 之上实时渲染姿态驱动的 3D 高斯溅射（3DGS）化身。

> 与 [webavatar-rust](https://github.com/aaaaaaaaaaaxi/webavatar-rust) WebGPU 浏览器端查看器配套。`.npz` / `.json` 资源相同，运行时不同。

## 论文

> **High-Fidelity Mobile Avatars with Pruned Local Blendshapes — WebGPU Viewer**
> Youyi Zhan, He Wang, Tianjia Shao, Kun Zhou · CVPR 2026
> [项目主页](https://gapszju.github.io/webavatar) · [arXiv](https://arxiv.org/abs/2605.01854)

## 特性

- **拖放即用** — 将 `.npz` 化身模型和 `.pose.json` 姿态序列放入 `Assets/`，自动导入为 `WebAvatarAsset` / `WebAvatarPoseAsset`。
- **姿态驱动 3DGS 渲染** — 每帧 compute 流水线执行 LBS 蒙皮、MLP 特征生成、属性混合形状和重心坐标变换；最终由 gsplat-unity 渲染器光栅化。
- **编辑模式预览** — 通过 `PreviewFrame` 滑块在 Scene 视图中检查单帧。
- **运行时回放** — `Play / Pause`、FPS 控制，以及 Loop / Clamp / Once 三种播完行为。
- **无需编译步骤** — 纯 UPM 包，无 DLL，无原生插件。

## 安装

### 1. 安装基础 gsplat 渲染器

按照 [`wuyize25/gsplat-unity`](https://github.com/wuyize25/gsplat-unity) 的 README 添加基础渲染器包。

### 2. 安装本包

在 `Packages/manifest.json` 中以本地 UPM 包形式引用本目录：

```json
// Packages/manifest.json
{
  "dependencies": {
    "org.webavatar.runtime": "file:../path/to/GSAvatar-unity",
    "wu.yize.gsplat": "1.1.2"
  }
}
```

### 3. 放入资源

将 `.npz` 和 `.pose.json` 文件复制到 `Assets/` 下任意位置（**不要**放在 `Resources/` 下）。

## 资源格式

| 文件 | 扩展名 | 生成的资源 | 来源 |
|---|---|---|---|
| 化身模型 | `.npz` | `WebAvatarAsset` | Python 训练代码 → 导出 NPZ |
| 姿态序列 | `.pose.json` | `WebAvatarPoseAsset` | Rust/WebGPU 查看器流水线 |

> 姿态 JSON 文件必须使用两段扩展名 `.pose.json`，因为 Unity 自带的 `TextScriptImporter` 已经占用了裸 `.json` 扩展名，否则会拒绝我们的 `ScriptedImporter` 注册。

## 使用

### 场景搭建

1. 在场景中创建一个空 GameObject。
2. 添加 **`WebAvatar/Web Avatar Renderer`** 组件（会自动添加 `WebAvatarComponent`）。
3. 在 Inspector 中指定已导入的 `WebAvatarAsset`（以及可选的 `WebAvatarPoseAsset`）。
4. 进入 Play 模式。

### Inspector 字段

| 字段 | 说明 |
|---|---|
| `Asset` | 已导入的 `.npz` 化身模型。 |
| `PoseAsset` | 可选的 `.pose.json` 姿态序列。留空则渲染规范（大）姿态。 |
| `Fps` | 姿态序列的播放速率。 |
| `PlayMode` | 播完行为：`Loop` / `Clamp` / `Once`。 |
| `Play` | 关闭后冻结在当前姿态，但仍继续渲染。 |
| `SHDegree` | gsplat 着色器使用的 SH 频段。随包发布的 NPZ 仅含频段 0，保持 `0` 即可。 |
| `GammaToLinear` | 在着色前将 gamma 空间颜色转换为线性空间。 |
| `PreviewFrame`（同级组件） | 非 Play 模式下 Scene 视图 gizmo 显示的帧索引。 |

### 代码访问

```csharp
using WebAvatar;
using WebAvatar.Pose;

var avatar  = GetComponent<WebAvatarComponent>();
var frame   = avatar.GetPose(42);        // PoseFrame? — 越界返回 null
int poses   = avatar.PoseCount;          // PoseAsset 未设置时为 0
bool loaded = avatar.HasAsset;
```

## 架构

```
WebAvatarAsset (.npz)  ─┐
                        ├─► WebAvatarComponent ─► WebAvatarRenderer ─► wu.yize.gsplat
WebAvatarPoseAsset      ─┘             │
                                       ▼
                  WebAvatarGpuResources（ComputeBuffer 池）
                                       │
                                       ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │  每帧 compute 流水线                                                 │
   │                                                                      │
   │  1. LbsSkinning.compute        — 按 G 关节蒙皮规范 splat             │
   │  2. MlpForward.compute         — 每部分特征生成器                    │
   │  3. AttributeBasis.compute     — 加载每 splat 的基系数               │
   │  4. AttributeDxyz.compute       — 解码 dxyz（位置偏移）              │
   │  5. AttributeTransform.compute — 应用旋转/缩放/颜色增量              │
   └─────────────────────────────────────────────────────────────────────┘
```

- **`Runtime/Avatar/`** — `ScriptableObject` 资源（`WebAvatarAsset`、`WebAvatarPoseAsset`）、渲染器 MonoBehaviour、GPU 资源池以及 CPU 端 SMPL-X 蒙皮数学。
- **`Runtime/Npz/`** — 极简 NPZ 阅读器（NumPy `.npy` 格式）。
- **`Runtime/Pose/`** — `PoseFrame` 结构体与 `PosePlayer` 序列播放器。
- **`Runtime/Resources/`** — 实现每帧流水线的 5 个 compute shader。
- **`Editor/`** — `.npz` 和 `.pose.json` 的 `ScriptedImporter`、自定义 Inspector 与调试菜单。

## 限制

- NPZ 中只发布了 SH 频段 0，因此 `SHDegree` 必须保持为 `0`。
- 姿态 JSON 必须重命名为 `.pose.json`（两段扩展名），以绕过 Unity 自带的 JSON 导入器。
- 导入时 float16 源数据会扩展为 float32；这会使规范缓冲区所占 GPU 内存约翻倍。

## 许可证

见 [webavatar-rust](https://github.com/aaaaaaaaaaaxi/webavatar-rust) 父仓库的 `LICENSE`。

## 引用

```bibtex
@inproceedings{zhan2026webavatar,
  title  = {High-Fidelity Mobile Avatars with Pruned Local Blendshapes},
  author = {Zhan, Youyi and Wang, He and Shao, Tianjia and Zhou, Kun},
  booktitle = {CVPR},
  year   = {2026}
}
```