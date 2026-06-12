// WebAvatarRenderer.cs — pose-driven runtime renderer for WebAvatar.
//
// The MonoBehaviour is also a Gsplat.IGsplat so that GsplatSorter picks
// it up and sorts its (position, scale, rotation, color) GraphicsBuffers
// each frame.  We fill those buffers from a per-frame compute pipeline
// (LBS + MLP + attribute blend shapes + barycentric transform) hosted
// in WebAvatarGpuResources, then hand the buffers to
// GsplatRendererImpl.Render() which rasterises them.
//
// Lifecycle: OnEnable / OnDisable mirrors GsplatRenderer — register
// with the sorter, build all GPU resources, mirror to canonical data,
// and tear it all down on disable.  The actual pose evaluation runs
// in Update so the camera's onPreCull hook sees a stable set of
// splats by the time the sorter dispatches.

using System;
using Gsplat;
using UnityEngine;
using WebAvatar.Pose;
using PlayMode = WebAvatar.Pose.PlayMode;

namespace WebAvatar
{
    [AddComponentMenu("WebAvatar/Web Avatar Renderer")]
    [ExecuteAlways]
    [DisallowMultipleComponent]
    public class WebAvatarRenderer : MonoBehaviour, IGsplat
    {
        [Tooltip("Imported .npz asset.")]
        public WebAvatarAsset Asset;

        [Tooltip("Optional .json pose sequence. Leave null for canonical pose.")]
        public WebAvatarPoseAsset PoseAsset;

        [Tooltip("Animation FPS for PoseAsset playback.")]
        public float Fps = 30f;

        [Tooltip("End-of-sequence behaviour: Loop / Clamp / Once.")]
        public PlayMode PlayMode = PlayMode.Loop;

        [Tooltip("Disable to pause pose playback while still rendering the canonical frame.")]
        public bool Play = true;

        [Tooltip("SH degree to use for the gsplat shader.  This avatar's NPZ only ships band 0 so leave at 0.")]
        [Range(0, 3)] public int SHDegree = 0;

        [Tooltip("If true, the rendered colors are in gamma space and will be converted to linear before shading.")]
        public bool GammaToLinear = false;

        // ---- Runtime state ----
        GsplatRendererImpl _gsplat;
        WebAvatarGpuResources _gpu;
        PosePlayer _player;
        WebAvatarComponent _component; // sibling, for PreviewFrame in Edit mode
        Matrix4x4[] _jointMats;        // length 55
        Matrix4x4[] _aCanonicalInv;    // cached
        float[] _jointMatsFlat;        // 55 * 16 floats (column-major)
        float[] _bigPose;              // 165 floats (rest pose = bigpose)

        // ---- IGsplat ----
        // transform / isActiveAndEnabled are inherited from MonoBehaviour
        public uint SplatCount => Asset != null ? (uint)Asset.SplatCount : 0;
        public ISorterResource SorterResource => _gsplat?.SorterResource;
        public bool Valid => Asset != null && _gsplat != null && _gpu != null;

        void OnEnable()
        {
            // Always cache the sibling — even if our own Asset/PoseAsset
            // fields are set, Update still needs the sibling's PreviewFrame
            // slider for Edit-mode preview.
            _component = GetComponent<WebAvatarComponent>();
            if (Asset == null && _component != null)
            {
                Asset = _component.Asset;
                if (PoseAsset == null) PoseAsset = _component.PoseAsset;
            }
            if (Asset == null) return;

            GsplatSorter.Instance.RegisterGsplat(this);
            _gsplat = new GsplatRendererImpl((uint)Asset.SplatCount, shBands: 0);
            _gpu = new WebAvatarGpuResources(Asset);
            _player = new PosePlayer
            {
                Frames = PoseAsset != null && PoseAsset.Frames != null
                    ? PoseAsset.Frames
                    : Array.Empty<PoseFrame>(),
                Fps = Fps,
                Mode = PlayMode,
            };
            // Pre-compute A_canonical_inv for the SMPL-X skinning.
            _jointMats = new Matrix4x4[WebAvatarSkinning.SMPL_NUM_JOINTS];
            _aCanonicalInv = new Matrix4x4[WebAvatarSkinning.SMPL_NUM_JOINTS];
            _jointMatsFlat = new float[WebAvatarSkinning.SMPL_NUM_JOINTS * 16];
            WebAvatarSkinning.ComputeCanonicalJointMatsInv(Asset.Joints, _aCanonicalInv);
            // The "big pose" (hip rotations ±25°) is the rest pose, so
            // when no pose asset is attached we render the avatar in its
            // canonical A-pose.
            _bigPose = new float[WebAvatarSkinning.SMPL_POSE_DIM];
            WebAvatarSkinning.InitSmplPose(_bigPose);
        }

        void OnDisable()
        {
            GsplatSorter.Instance.UnregisterGsplat(this);
            _gsplat?.Dispose();
            _gsplat = null;
            _gpu?.Dispose();
            _gpu = null;
            _component = null;
        }

        void Update()
        {
            if (Asset == null || _gsplat == null || _gpu == null) return;
            if (_gsplat.PositionBuffer == null) return;

            // Re-sync any inspector-tweakable fields onto the player every
            // tick so users can change Fps / PlayMode / PoseAsset at runtime
            // (or while paused in Edit mode) without having to toggle the
            // component off and on.  All three writes are cheap; Frames is
            // a reference assignment, Fps/Mode are value-type writes.
            SyncPlayer();

            // 1) Select the pose frame for this tick.
            //    Play mode  : drive off Time.time so the sequence plays at FPS.
            //    Edit mode  : drive off the sibling's PreviewFrame slider so
            //                 the SceneView shows a deterministic frame rather
            //                 than jittering on Editor repaint events
            //                 (Time.time in Edit returns timeSinceStartup, which
            //                 is huge and only advances when Update fires).
            PoseFrame pose;
            if (_player == null)
            {
                pose = default;
            }
            else if (Application.isPlaying)
            {
                float t = Play ? Time.time : 0f;
                pose = _player.Evaluate(t);
            }
            else
            {
                int idx = _component != null ? _component.PreviewFrame : 0;
                pose = _player.GetFrame(idx);
            }

            // 2) Compute skinning matrices G on the CPU.  When no pose
            //    asset is attached, fall back to the rest (big) pose so
            //    the avatar still shows up in canonical T/A-pose.
            var poseVec = (pose.pose != null && pose.pose.Length == WebAvatarSkinning.SMPL_POSE_DIM)
                ? pose.pose
                : _bigPose;
            WebAvatarSkinning.ComputePoseJointMats(
                poseVec, Asset.Joints, _aCanonicalInv, _jointMats);
            FlattenJointMats(_jointMats, _jointMatsFlat);

            // 3) Build the global rigid transform (transl + rot).
            Matrix4x4 globalMat = WebAvatarSkinning.BuildGlobalMatrix(
                pose.transl != null && pose.transl.Length == 3
                    ? new Vector3(pose.transl[0], pose.transl[1], pose.transl[2])
                    : Vector3.zero,
                pose.rotation != null && pose.rotation.Length == 9
                    ? pose.rotation
                    : IdentityRotationRowMajor);

            // 4) Dispatch the full GPU pipeline.
            _gpu.DispatchFrame(pose, _jointMatsFlat, globalMat,
                _gsplat.PositionBuffer, _gsplat.ScaleBuffer,
                _gsplat.RotationBuffer, _gsplat.ColorBuffer);

            // 5) Hand the buffers to the gsplat renderer.  The sorter
            //    runs in Camera.onPreCull and reads the same buffers.
            _gsplat.Render(SplatCount, transform, Asset.Bounds,
                gameObject.layer, GammaToLinear, SHDegree);
        }

        // --- helpers ---

        // Mirror the Inspector-visible Fps / PlayMode / PoseAsset onto the
        // cached PosePlayer so live tweaks take effect on the next tick.
        // Swapping the actual WebAvatarAsset is NOT handled here — that
        // would require rebuilding _gsplat / _gpu (different splat count,
        // different MLP layers) and is handled via OnDisable/OnEnable.
        void SyncPlayer()
        {
            if (_player == null) return;
            var frames = PoseAsset != null && PoseAsset.Frames != null
                ? PoseAsset.Frames
                : Array.Empty<PoseFrame>();
            _player.Frames = frames;
            _player.Fps = Fps;
            _player.Mode = PlayMode;
        }

        static readonly float[] IdentityRotationRowMajor = new float[9]
        {
            1, 0, 0,
            0, 1, 0,
            0, 0, 1,
        };

        static void FlattenJointMats(Matrix4x4[] mats, float[] dst)
        {
            // Each joint matrix is uploaded as 4 float4 entries, one per
            // column, matching the HLSL `load_joint_mat`:
            //   float4x4 M = float4x4(_JointMats[i*4+0],
            //                          _JointMats[i*4+1],
            //                          _JointMats[i*4+2],
            //                          _JointMats[i*4+3]);
            // The float4-constructor in HLSL takes columns, so we
            // explicitly walk the Matrix4x4 by GetColumn(c) — Unity's
            // storage is column-major but the natural `m.m00, m.m01, ...`
            // enumeration is row-major.
            for (int i = 0; i < mats.Length; i++)
            {
                int o = i * 16;
                var m = mats[i];
                var c0 = m.GetColumn(0);
                var c1 = m.GetColumn(1);
                var c2 = m.GetColumn(2);
                var c3 = m.GetColumn(3);
                dst[o +  0] = c0.x; dst[o +  1] = c0.y; dst[o +  2] = c0.z; dst[o +  3] = c0.w;
                dst[o +  4] = c1.x; dst[o +  5] = c1.y; dst[o +  6] = c1.z; dst[o +  7] = c1.w;
                dst[o +  8] = c2.x; dst[o +  9] = c2.y; dst[o + 10] = c2.z; dst[o + 11] = c2.w;
                dst[o + 12] = c3.x; dst[o + 13] = c3.y; dst[o + 14] = c3.z; dst[o + 15] = c3.w;
            }
        }
    }
}
