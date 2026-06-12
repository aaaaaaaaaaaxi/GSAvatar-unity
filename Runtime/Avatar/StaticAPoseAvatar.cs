// StaticAPoseAvatar.cs — minimal A-pose renderer wrapper.
//
// Wraps a single WebAvatarAsset reference and forwards it to the
// auto-attached WebAvatarRenderer.  No pose asset is wired up, so
// the renderer falls back to its rest ("big") pose — the canonical
// A-pose that the webavatar-rust pipeline uses as the resting
// configuration (see WebAvatarSkinning.InitSmplPose).
//
// The component is intentionally a thin shim: all the LBS / MLP /
// attribute math is unchanged, and lives in WebAvatarSkinning,
// WebAvatarGpuResources and the existing WebAvatarRenderer.

using UnityEngine;

namespace WebAvatar
{
    [AddComponentMenu("WebAvatar/Static A-Pose Avatar")]
    [ExecuteAlways]
    [DisallowMultipleComponent]
    [RequireComponent(typeof(WebAvatarRenderer))]
    public class StaticAPoseAvatar : MonoBehaviour
    {
        [Tooltip("Imported .npz asset (drag from the Project window).")]
        public WebAvatarAsset Asset;

        void OnEnable()
        {
            var renderer = GetComponent<WebAvatarRenderer>();
            if (renderer == null) return;
            // Forward the asset.  Leaving PoseAsset null on the renderer
            // is what makes the avatar draw in its canonical A-pose.
            renderer.Asset = Asset;
        }

        void OnValidate()
        {
            // Inspector edits should take effect immediately so the
            // SceneView reflects the new asset without a play-mode
            // toggle.
            if (!isActiveAndEnabled) return;
            var renderer = GetComponent<WebAvatarRenderer>();
            if (renderer == null) return;
            renderer.Asset = Asset;
        }
    }
}
