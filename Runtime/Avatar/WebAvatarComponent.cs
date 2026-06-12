// WebAvatarComponent.cs — runtime entry point for a webavatar.
//
// Holds references to the imported canonical-data asset and the
// (optional) pose sequence, logs a one-line summary on enable, and
// exposes a few read-only properties for downstream systems.
//
// The actual GPU pipeline (LBS + MLP + attribute blend shapes +
// wu.yize.gsplat rasterisation) lives in the sibling
// `WebAvatarRenderer`, which is auto-attached via [RequireComponent].

using System;
using UnityEngine;
using WebAvatar.Pose;

namespace WebAvatar
{
    [AddComponentMenu("WebAvatar/Web Avatar Component")]
    [ExecuteAlways]
    [RequireComponent(typeof(WebAvatarRenderer))]
    public class WebAvatarComponent : MonoBehaviour
    {
        [Tooltip("Imported .npz asset.")]
        public WebAvatarAsset Asset;

        [Tooltip("Optional .json pose sequence.")]
        public WebAvatarPoseAsset PoseAsset;

        [Tooltip("Frame index to preview in the scene-view gizmo.")]
        public int PreviewFrame;

        public bool HasAsset => Asset != null;
        public bool HasPose => PoseAsset != null && PoseAsset.Frames != null;

        public int PoseCount => HasPose ? PoseAsset.Frames.Length : 0;

        public PoseFrame? GetPose(int index)
        {
            if (!HasPose) return null;
            if (index < 0 || index >= PoseAsset.Frames.Length) return null;
            return PoseAsset.Frames[index];
        }

        void OnEnable()
        {
            if (Asset == null) return;
            Debug.Log(
                $"[WebAvatar] Loaded '{Asset.name}': splats={Asset.SplatCount}, " +
                $"joints={Asset.JointCount}, pruned={Asset.PrunedSplatCount}, " +
                $"ctrlPts={Asset.ControlPointCount}, mlpLayers={(Asset.MlpLayers == null ? 0 : Asset.MlpLayers.Length)}, " +
                $"headParts={Asset.NumHeadParts}, " +
                $"poses={(PoseAsset == null ? 0 : PoseAsset.FrameCount)}",
                this);
        }
    }
}
