// PoseFrame.cs — data shapes for a single avatar pose.
//
// One frame is what a SMPL-X driven avatar expects per step:
//   pose        165 floats  axis-angle for 55 joints (3 * 55)
//   Th          3 floats    root translation
//   Rh          9 floats    3x3 root rotation, row-major
//   expression  10 floats   FLAME expression (optional, nullable in source)
//
// We also store a frame_id that the JSON source provides, so the pose
// sequence can be sparse / non-monotonic if needed.

using System;
using System.Text.RegularExpressions;
using UnityEngine;

namespace WebAvatar.Pose
{
    [Serializable]
    public struct PoseFrame
    {
        public int frameId;
        public float[] pose;        // length 165 (SMPL-X 55 joints * 3)
        public float[] transl;      // length 3
        public float[] rotation;    // length 9, row-major 3x3
        public float[] expression;  // length 10, may be null
    }

    [Serializable]
    public class PoseJsonFrame
    {
        public int frame_id;
        public float[] pose;
        public float[] Th;
        public float[] Rh;
        public float[] expression;
    }

    // JsonUtility cannot deserialize a top-level array, so we wrap the
    // file's array into an object on the way in.
    [Serializable]
    class PoseJsonWrapper
    {
        public PoseJsonFrame[] frames;
    }

    public static class PoseJson
    {
        // Public entry point: feed the raw JSON text, get back a flat array
        // of PoseFrame with consistent sizes and missing expressions filled
        // with zeros.  Malformed entries are skipped (and reported via
        // warnings) rather than aborting the whole import.
        public static PoseFrame[] Parse(string json)
        {
            if (string.IsNullOrEmpty(json)) return Array.Empty<PoseFrame>();
            // The Rust loader reads Rh as [[f32;3];3] (nested 3x3 matrix) and
            // that is how the dataset is exported. Unity's JsonUtility cannot
            // deserialize a nested array into a float[] — it silently reads
            // the *outer* length and discards the inner values. We pre-flatten
            // the "Rh": [[...],[...],[...]] form to "Rh": [...] (9 floats)
            // before handing off to JsonUtility.
            json = FlattenNestedRh(json);
            // JsonUtility needs an object wrapper, not a top-level array.
            string wrapped = "{\"frames\":" + json + "}";

            PoseJsonWrapper wrapper;
            try
            {
                wrapper = JsonUtility.FromJson<PoseJsonWrapper>(wrapped);
            }
            catch (Exception e)
            {
                throw new InvalidOperationException("Failed to parse pose JSON: " + e.Message, e);
            }
            if (wrapper?.frames == null)
                throw new InvalidOperationException("Pose JSON contained no frames");

            var result = new PoseFrame[wrapper.frames.Length];
            for (int i = 0; i < wrapper.frames.Length; i++)
                result[i] = Convert(wrapper.frames[i], i);
            return result;
        }

        static PoseFrame Convert(PoseJsonFrame src, int index)
        {
            var dst = new PoseFrame();
            dst.frameId = src.frame_id;

            if (src.pose == null)
                throw new InvalidOperationException($"Frame {index}: missing 'pose'");
            if (src.pose.Length != 165)
                Debug.LogWarning(
                    $"Pose frame {index}: expected 165 pose params, got {src.pose.Length}");
            dst.pose = (float[])src.pose.Clone();

            if (src.Th == null || src.Th.Length != 3)
                throw new InvalidOperationException(
                    $"Frame {index}: 'Th' must be 3 floats, got " +
                    (src.Th == null ? "null" : src.Th.Length.ToString()));
            dst.transl = new float[3];
            Array.Copy(src.Th, dst.transl, 3);

            if (src.Rh == null)
                throw new InvalidOperationException($"Frame {index}: missing 'Rh'");
            // After FlattenNestedRh, Rh must be the 9-float row-major form.
            // Anything else means the source had an unexpected shape.
            if (src.Rh.Length != 9)
            {
                throw new InvalidOperationException(
                    $"Frame {index}: 'Rh' has unexpected length {src.Rh.Length} (expected 9)");
            }
            dst.rotation = (float[])src.Rh.Clone();

            if (src.expression != null)
            {
                if (src.expression.Length != 10)
                    Debug.LogWarning(
                        $"Pose frame {index}: expected 10 expression coeffs, got {src.expression.Length}");
                dst.expression = (float[])src.expression.Clone();
            }
            return dst;
        }

        // Convert "Rh": [[a,b,c],[d,e,f],[g,h,i]]  →  "Rh": [a,b,c,d,e,f,g,h,i].
        // The inner-array tokens never contain ']', so a non-greedy "[^\\]]*"
        // match is safe for the numbers our exporter writes (plain decimals,
        // optional sign / scientific notation).  We leave already-flat Rh
        // arrays untouched because they will not match the "[[" pattern.
        static readonly Regex s_NestedRh = new Regex(
            "\"Rh\"\\s*:\\s*\\[\\s*\\[([^\\]]*)\\]\\s*,\\s*\\[([^\\]]*)\\]\\s*,\\s*\\[([^\\]]*)\\]\\s*\\]",
            RegexOptions.Compiled);

        static string FlattenNestedRh(string json)
        {
            return s_NestedRh.Replace(json, m =>
            {
                string a = m.Groups[1].Value.Trim();
                string b = m.Groups[2].Value.Trim();
                string c = m.Groups[3].Value.Trim();
                return "\"Rh\": [" + a + ", " + b + ", " + c + "]";
            });
        }
    }
}
