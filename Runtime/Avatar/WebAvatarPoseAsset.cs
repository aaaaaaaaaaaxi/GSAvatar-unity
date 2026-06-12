// WebAvatarPoseAsset.cs — pose sequence produced by importing the JSON
// file that accompanies a webavatar-rust NPZ.  Each frame is the same
// shape the LBS / MLP / attribute shaders expect.

using UnityEngine;
using WebAvatar.Pose;

namespace WebAvatar
{
    public class WebAvatarPoseAsset : ScriptableObject
    {
        [Header("Pose sequence")]
        public int FrameCount;
        public PoseFrame[] Frames;
    }
}
