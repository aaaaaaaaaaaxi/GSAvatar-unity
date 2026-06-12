// AvatarDataBuilder.cs — pure NPZ -> WebAvatarAsset conversion.
//
// This is split out from the ScriptedImporter so the same logic can be
// exercised by tests / batch tools without needing the asset database.
// All float16 source data is widened to float32; all byte/uint buffers
// are left at their native width.

using System;
using System.Collections.Generic;
using UnityEngine;
using WebAvatar;
using WebAvatar.Npz;

namespace WebAvatar.Editor
{
    public static class AvatarDataBuilder
    {
        public sealed class BuildException : Exception
        {
            public BuildException(string msg) : base(msg) { }
        }

        // Build a populated WebAvatarAsset from a path on disk.
        public static WebAvatarAsset Build(string npzPath)
        {
            using var archive = NpzArchive.Open(npzPath);
            return Build(archive);
        }

        public static WebAvatarAsset Build(NpzArchive archive)
        {
            var asset = ScriptableObject.CreateInstance<WebAvatarAsset>();
            try
            {
                // ---------- Gaussians ----------
                // _xyz, _scaling, _rotation, _sh0, opacity
                var xyz     = Require(archive, "_xyz",     -1, 3).ToFloat32();
                var scaling = Require(archive, "_scaling", -1, 3).ToFloat32();
                var rot     = Require(archive, "_rotation",-1, 4).ToFloat32();
                var sh0     = Require(archive, "_sh0",     -1, 1, 3).ToFloat32();
                var opacity = Require(archive, "opacity",  -1).AsUInt8().ToArray();

                int splatCount = xyz.Length / 3;
                if (scaling.Length / 3 != splatCount ||
                    rot.Length / 4 != splatCount ||
                    sh0.Length / 3 != splatCount ||
                    opacity.Length != splatCount)
                {
                    throw new BuildException(
                        $"Splat count mismatch: xyz={xyz.Length/3}, scaling={scaling.Length/3}, " +
                        $"rot={rot.Length/4}, sh0={sh0.Length/3}, opacity={opacity.Length}");
                }

                asset.SplatCount = splatCount;
                asset.Positions = new Vector3[splatCount];
                asset.Scales    = new Vector3[splatCount];
                asset.Rotations = new Vector4[splatCount];
                asset.Sh0       = new Color[splatCount];
                asset.Opacities = new float[splatCount];

                var bounds = new Bounds();
                for (int i = 0; i < splatCount; i++)
                {
                    asset.Positions[i] = new Vector3(xyz[3*i], xyz[3*i+1], xyz[3*i+2]);
                    asset.Scales[i]    = new Vector3(scaling[3*i], scaling[3*i+1], scaling[3*i+2]);
                    var q = new Vector4(rot[4*i], rot[4*i+1], rot[4*i+2], rot[4*i+3]);
                    // Normalise defensively — gsplat-unity expects a unit quat.
                    float qLen = Mathf.Sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);
                    if (qLen > 1e-8f) q /= qLen;
                    asset.Rotations[i] = q;
                    // SH band 0 is the constant (DC) term.  Converting back to
                    // a visible RGB uses the standard 0.5 + 0.5*Y00 hack; the
                    // WebAvatar pipeline only ever ships SH band 0, so this
                    // matches what the Rust viewer does.
                    asset.Sh0[i] = new Color(
                        sh0[3*i]   + 0.5f,
                        sh0[3*i+1] + 0.5f,
                        sh0[3*i+2] + 0.5f,
                        1f);
                    asset.Opacities[i] = opacity[i] / 255f;
                    if (i == 0) bounds = new Bounds(asset.Positions[i], Vector3.zero);
                    else bounds.Encapsulate(asset.Positions[i]);
                }
                asset.Bounds = bounds;

                // ---------- LBS ----------
                var joints     = Require(archive, "t_joints",    -1, 3).ToFloat32();
                var weightsIdx = Require(archive, "weights_idx", -1, 4).AsUInt8().ToArray();
                var weightsVal = Require(archive, "weights_val", -1, 4).AsUInt8().ToArray();

                int jointCount = joints.Length / 3;
                if (weightsIdx.Length != splatCount * 4 ||
                    weightsVal.Length != splatCount * 4)
                {
                    throw new BuildException(
                        $"LBS weights size mismatch: idx={weightsIdx.Length}, " +
                        $"val={weightsVal.Length}, expected {splatCount*4}");
                }
                asset.JointCount = jointCount;
                asset.Joints     = new Vector3[jointCount];
                for (int j = 0; j < jointCount; j++)
                    asset.Joints[j] = new Vector3(joints[3*j], joints[3*j+1], joints[3*j+2]);
                asset.WeightsIdx = weightsIdx;
                asset.WeightsVal = weightsVal;

                // ---------- MLP ----------
                if (archive.TryGetArray("num_xyz_ft_head", out var nfh))
                {
                    if (nfh.Header.NumElements != 1 || nfh.Header.DType != NpyDType.UInt32)
                        throw new BuildException(
                            $"num_xyz_ft_head must be uint32 scalar, got dtype={nfh.Header.DType} " +
                            $"nelements={nfh.Header.NumElements}");
                    asset.NumHeadParts = (int)nfh.ToUInt32()[0];
                }
                asset.MlpLayers = BuildMlp(archive);

                // ---------- Attribute ----------
                if (archive.TryGetArray("gs_feat_knn", out var gsFeatKnn))
                    asset.GsFeatKnn = gsFeatKnn.ToUInt16();

                if (archive.TryGetArray("rot_idxs", out var rotIdxs))
                {
                    asset.RotIdxs = rotIdxs.ToUInt32();
                    asset.PrunedSplatCount = asset.RotIdxs.Length;
                }
                if (archive.TryGetArray("scale_idxs", out var scaleIdxs))
                    asset.ScaleIdxs = scaleIdxs.ToUInt32();
                if (archive.TryGetArray("color_idxs", out var colorIdxs))
                    asset.ColorIdxs = colorIdxs.ToUInt32();

                asset.ColorBasis    = BuildBasis(archive, "color_basis",    asset.PrunedSplatCount, 3);
                asset.ScaleBasis    = BuildBasis(archive, "scaling_basis",  asset.PrunedSplatCount, 3);
                asset.RotationBasis = BuildBasis(archive, "rotation_basis", asset.PrunedSplatCount, 4);

                if (archive.TryGetArray("ctrl_pt_offsets", out var cpo))
                {
                    var cpoF = cpo.ToFloat32();
                    asset.ControlPointCount = cpoF.Length / 3;
                    asset.ControlPointOffsets = cpoF;
                }
                asset.ControlPointBasis = BuildBasis(archive, "ctrl_pt_basis", asset.ControlPointCount, 3);

                if (archive.TryGetArray("gs_ctrl_knn", out var gck))
                    asset.GsCtrlKnn = gck.ToUInt16();
                if (archive.TryGetArray("gs_ctrl_knn_weights", out var gckw))
                {
                    asset.GsCtrlKnnWeights = gckw.ToFloat32();
                }
                if (archive.TryGetArray("ctrl_feat_knn", out var cfk))
                    asset.CtrlFeatKnn = cfk.ToUInt16();

                return asset;
            }
            catch
            {
                UnityEngine.Object.DestroyImmediate(asset);
                throw;
            }
        }

        // Build a [N, dim, 16] basis into a flat float[] of length N*dim*16.
        // N is derived from the array's own element count; expectedCount
        // is just a consistency check (0 to skip).
        static float[] BuildBasis(NpzArchive archive, string name, int expectedCount, int dim)
        {
            if (!archive.TryGetArray(name, out var arr)) return Array.Empty<float>();
            int stride = dim * WebAvatarAsset.BasisComponents;
            int countFromData = arr.Header.NumElements / stride;
            if (expectedCount > 0 && countFromData != expectedCount)
                throw new BuildException(
                    $"Basis '{name}' has {countFromData} entries, expected {expectedCount}");
            return arr.ToFloat32();
        }

        // Enumerate layers.{i}.weight / .bias for even i starting at 0.
        // The webavatar-rust pipeline writes a final activation of 0 (no
        // activation) on the last layer; we reproduce that flag.
        static WebAvatarAsset.MlpLayerData[] BuildMlp(NpzArchive archive)
        {
            var result = new List<WebAvatarAsset.MlpLayerData>();
            for (int i = 0; ; i += 2)
            {
                string wName = $"layers.{i}.weight";
                string bName = $"layers.{i}.bias";
                if (!archive.Contains(wName)) break;

                var wArr = archive.GetArray(wName);
                var bArr = archive.GetArray(bName);

                var layer = new WebAvatarAsset.MlpLayerData
                {
                    WeightShape = wArr.Header.Shape,
                    Weights = wArr.ToFloat32(),
                    Biases = bArr.ToFloat32(),
                };
                // wArr shape is [Out, Partition, In] for layer 0 and [Out, In] otherwise
                if (wArr.Header.Shape.Length == 3)
                {
                    layer.OutFeatures = (int)wArr.Header.Shape[0];
                    layer.InFeatures  = (int)wArr.Header.Shape[2];
                }
                else if (wArr.Header.Shape.Length == 2)
                {
                    layer.OutFeatures = (int)wArr.Header.Shape[0];
                    layer.InFeatures  = (int)wArr.Header.Shape[1];
                }
                else
                {
                    throw new BuildException(
                        $"Unexpected MLP weight rank {wArr.Header.Shape.Length} for {wName}");
                }
                layer.Activation = 1;
                result.Add(layer);
            }
            // Mark the last layer as activation=0 (no activation), matching Rust.
            if (result.Count > 0)
                result[result.Count - 1].Activation = 0;
            return result.ToArray();
        }

        // Require an array to exist with a specific shape.  Pass -1 in any
        // dimension to wildcard it.
        static NpyArray Require(NpzArchive archive, string name, params int[] shape)
        {
            var arr = archive.GetArray(name);
            var actual = arr.Header.Shape;
            if (actual.Length != shape.Length)
                throw new BuildException(
                    $"Array '{name}' rank {actual.Length} != expected {shape.Length}");
            for (int i = 0; i < actual.Length; i++)
            {
                if (shape[i] < 0) continue;
                if (actual[i] != shape[i])
                    throw new BuildException(
                        $"Array '{name}' dim {i} is {actual[i]}, expected {shape[i]}");
            }
            return arr;
        }
    }
}
