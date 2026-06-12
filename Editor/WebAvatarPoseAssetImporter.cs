// WebAvatarPoseAssetImporter.cs — turns a .json pose file into a
// WebAvatarPoseAsset.  The JSON must be a top-level array of frame
// objects (see PoseFrame.cs for the schema).

using System.IO;
using UnityEditor.AssetImporters;
using UnityEngine;
using WebAvatar.Pose;

namespace WebAvatar.Editor
{
    // We register under ".pose.json" (two-segment extension) instead of
    // ".json" because Unity's native TextScriptImporter already owns the
    // bare ".json" extension and ScriptedImporter registration is rejected
    // when a native importer covers the same extension.  Renaming pose
    // files to "foo.pose.json" lets the asset DB route them to us without
    // the registration error.
    [ScriptedImporter(1, "pose.json")]
    public class WebAvatarPoseAssetImporter : ScriptedImporter
    {
        // The file must look like our schema: a top-level array containing
        // "pose" keys.  Anything else is rejected so a stray ".pose.json"
        // file with unrelated content still surfaces an error rather than
        // being silently turned into an empty pose asset.
        public override void OnImportAsset(AssetImportContext ctx)
        {
            string text = File.ReadAllText(ctx.assetPath);
            if (!LooksLikePoseJson(text))
                throw new System.InvalidOperationException(
                    "Not a webavatar pose JSON (expected a top-level array of " +
                    "frame objects with a 'pose' field): " + ctx.assetPath);

            var frames = PoseJson.Parse(text);
            var asset = ScriptableObject.CreateInstance<WebAvatarPoseAsset>();
            asset.name = Path.GetFileNameWithoutExtension(ctx.assetPath);
            asset.FrameCount = frames.Length;
            asset.Frames = frames;

            ctx.AddObjectToAsset("WebAvatarPoseAsset", asset);
            ctx.SetMainObject(asset);
        }

        // Cheap check: must start with '[' and contain a "pose": token
        // within the first 4 KB.  Avoids hijacking generic JSON files.
        static bool LooksLikePoseJson(string text)
        {
            if (string.IsNullOrEmpty(text)) return false;
            int len = Mathf.Min(text.Length, 4096);
            string head = text.Substring(0, len);
            int firstNonWs = 0;
            while (firstNonWs < head.Length && char.IsWhiteSpace(head[firstNonWs])) firstNonWs++;
            if (firstNonWs >= head.Length || head[firstNonWs] != '[') return false;
            return head.Contains("\"pose\"");
        }
    }
}
