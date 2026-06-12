// PosePlayer.cs — time-based frame selection over a pose sequence.
//
// Drives a PoseFrame[] (typically loaded from WebAvatarPoseAsset) using a
// seconds-since-enable clock, with FPS-driven sample rate, plus loop /
// clamp end behaviour.  Lives next to PoseFrame so the rendering
// pipeline has a single, predictable way to query "the pose for time t".
//
// The renderer treats identity pose (zeros, zero transl, identity rot)
// as a valid input — PosePlayer.Evaluate returns that when Frames is
// null/empty so callers don't have to special-case the no-pose-asset
// path.

using UnityEngine;

namespace WebAvatar.Pose
{
    public enum PlayMode
    {
        Loop,    // wrap around to the first frame at the end
        Clamp,   // hold the last frame
        Once,    // hold the first frame after the sequence ends
    }

    public class PosePlayer
    {
        public PoseFrame[] Frames;
        public float Fps = 30f;
        public PlayMode Mode = PlayMode.Loop;

        static readonly PoseFrame s_identity = new PoseFrame
        {
            frameId = 0,
            pose = null,        // null pose is treated as zeros by the renderer
            transl = null,
            rotation = null,
            expression = null,
        };

        public int FrameCount => Frames != null ? Frames.Length : 0;

        // Look up a specific frame by index using the same end-of-sequence
        // behaviour as Evaluate(time).  Used by the editor preview path so a
        // user-facing PreviewFrame slider stays valid no matter what value
        // they drag (negative, past the end, etc.).
        public PoseFrame GetFrame(int index)
        {
            int n = FrameCount;
            if (n == 0) return s_identity;
            if (n == 1) return Frames[0];
            int idx;
            switch (Mode)
            {
                case PlayMode.Clamp:
                    idx = Mathf.Clamp(index, 0, n - 1);
                    break;
                case PlayMode.Once:
                    idx = Mathf.Min(Mathf.Max(index, 0), n - 1);
                    break;
                case PlayMode.Loop:
                default:
                    idx = ((index % n) + n) % n;
                    break;
            }
            return Frames[idx];
        }

        public PoseFrame Evaluate(float time)
        {
            int n = FrameCount;
            if (n == 0) return s_identity;
            if (n == 1) return Frames[0];

            int idx;
            float frame = time * Mathf.Max(1e-3f, Fps);
            switch (Mode)
            {
                case PlayMode.Clamp:
                    idx = Mathf.Clamp(Mathf.FloorToInt(frame), 0, n - 1);
                    break;
                case PlayMode.Once:
                    idx = Mathf.Min(Mathf.FloorToInt(frame), n - 1);
                    break;
                case PlayMode.Loop:
                default:
                    idx = ((Mathf.FloorToInt(frame) % n) + n) % n;
                    break;
            }
            return Frames[idx];
        }
    }
}
