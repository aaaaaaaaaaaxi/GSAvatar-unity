// WebAvatarDebugMenu.cs — Tools > WebAvatar menu items.
//
// These exist primarily to smoke-test the parsers without setting up a
// full Unity scene.  Both items show up under "Tools/WebAvatar".

using System.IO;
using UnityEditor;
using UnityEngine;
using WebAvatar.Pose;

namespace WebAvatar.Editor
{
    public static class WebAvatarDebugMenu
    {
        [MenuItem("Tools/WebAvatar/Inspect NPZ file…", priority = 100)]
        static void InspectNpz()
        {
            string path = EditorUtility.OpenFilePanel("Pick an .npz file", "", "npz");
            if (string.IsNullOrEmpty(path)) return;
            Debug.Log($"[WebAvatar] Inspecting NPZ: {path}");

            using var archive = Npz.NpzArchive.Open(path);
            int entries = 0;
            foreach (var name in archive.Names)
            {
                var arr = archive.GetArray(name);
                var shape = arr.Header.Shape == null || arr.Header.Shape.Length == 0
                    ? "()"
                    : "(" + string.Join(",", arr.Header.Shape) + ")";
                Debug.Log($"  {name}  dtype={arr.Header.DType}  shape={shape}  bytes={arr.Data.Length}");
                entries++;
                if (entries >= 30) { Debug.Log("  … (truncated, more in the file)"); break; }
            }
        }

        [MenuItem("Tools/WebAvatar/Inspect Pose JSON…", priority = 101)]
        static void InspectPose()
        {
            string path = EditorUtility.OpenFilePanel("Pick a pose .json file", "", "json");
            if (string.IsNullOrEmpty(path)) return;
            Debug.Log($"[WebAvatar] Inspecting pose JSON: {path}");

            var text = File.ReadAllText(path);
            var frames = PoseJson.Parse(text);
            Debug.Log($"  {frames.Length} frames");
            for (int i = 0; i < Mathf.Min(3, frames.Length); i++)
            {
                var f = frames[i];
                Debug.Log($"  frame {i}: id={f.frameId}  pose={f.pose?.Length}  " +
                          $"transl={(f.transl == null ? -1 : f.transl.Length)}  " +
                          $"rot={(f.rotation == null ? -1 : f.rotation.Length)}  " +
                          $"expr={(f.expression == null ? 0 : f.expression.Length)}");
            }
        }
    }
}
