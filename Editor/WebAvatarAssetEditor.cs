// WebAvatarAssetEditor.cs — inspector that shows a parsed-NPZ summary.
//
// Most fields are stored on the asset but hidden from the default
// inspector (they're huge float arrays).  This custom inspector
// surfaces the metadata plus a few sanity-check values so the user
// can confirm the import succeeded without re-opening the NPZ.

using UnityEditor;
using UnityEngine;

namespace WebAvatar.Editor
{
    [CustomEditor(typeof(WebAvatarAsset))]
    public class WebAvatarAssetEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            var asset = (WebAvatarAsset)target;
            serializedObject.Update();

            using (new EditorGUI.DisabledScope(true))
            {
                EditorGUILayout.LabelField("Canonical Gaussians", EditorStyles.boldLabel);
                EditorGUILayout.IntField("Splat count", asset.SplatCount);
                EditorGUILayout.IntField("Joint count", asset.JointCount);
                EditorGUILayout.IntField("Pruned splat count", asset.PrunedSplatCount);
                EditorGUILayout.IntField("Control point count", asset.ControlPointCount);
                EditorGUILayout.IntField("Head parts (MLP)", asset.NumHeadParts);
                EditorGUILayout.IntField("MLP layers",
                    asset.MlpLayers == null ? 0 : asset.MlpLayers.Length);

                BoundsField("Canonical AABB", asset.Bounds);

                EditorGUILayout.Space(6);
                EditorGUILayout.LabelField("First splat sample", EditorStyles.boldLabel);
                if (asset.Positions != null && asset.Positions.Length > 0)
                {
                    EditorGUILayout.Vector3Field("Position[0]", asset.Positions[0]);
                    EditorGUILayout.Vector3Field("Scale[0]",   asset.Scales[0]);
                    EditorGUILayout.Vector4Field("Rotation[0]", asset.Rotations[0]);
                    EditorGUILayout.ColorField("SH0[0]",        asset.Sh0[0]);
                    EditorGUILayout.FloatField("Opacity[0]",    asset.Opacities[0]);
                }

                if (asset.MlpLayers != null)
                {
                    EditorGUILayout.Space(6);
                    EditorGUILayout.LabelField("MLP layer shapes", EditorStyles.boldLabel);
                    foreach (var layer in asset.MlpLayers)
                    {
                        string shape = layer.WeightShape == null
                            ? "?"
                            : string.Join("x", layer.WeightShape);
                        EditorGUILayout.LabelField(
                            $"  in={layer.InFeatures}  out={layer.OutFeatures}  " +
                            $"act={layer.Activation}  weights=[{shape}]");
                    }
                }
            }
            serializedObject.ApplyModifiedProperties();
        }

        static void BoundsField(string label, Bounds b)
        {
            using (new EditorGUILayout.HorizontalScope())
            {
                EditorGUILayout.PrefixLabel(label);
                EditorGUILayout.LabelField(
                    $"center={b.center}  size={b.size}  min={b.min}  max={b.max}");
            }
        }
    }

    [CustomEditor(typeof(WebAvatarPoseAsset))]
    public class WebAvatarPoseAssetEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            var asset = (WebAvatarPoseAsset)target;
            serializedObject.Update();

            using (new EditorGUI.DisabledScope(true))
            {
                EditorGUILayout.IntField("Frame count", asset.FrameCount);
                if (asset.Frames != null && asset.Frames.Length > 0)
                {
                    var f0 = asset.Frames[0];
                    EditorGUILayout.IntField("First frame_id", f0.frameId);
                    EditorGUILayout.IntField("Pose length",
                        f0.pose == null ? 0 : f0.pose.Length);
                    EditorGUILayout.IntField("Expression length",
                        f0.expression == null ? 0 : f0.expression.Length);
                    if (f0.transl != null)
                        EditorGUILayout.Vector3Field("First Th", new Vector3(
                            f0.transl[0], f0.transl[1], f0.transl[2]));
                    if (f0.rotation != null)
                        EditorGUILayout.LabelField(
                            "First Rh (row-major)",
                            string.Format("[{0:F2} {1:F2} {2:F2}]",
                                f0.rotation[0], f0.rotation[1], f0.rotation[2]));
                }
            }
            serializedObject.ApplyModifiedProperties();
        }
    }
}
