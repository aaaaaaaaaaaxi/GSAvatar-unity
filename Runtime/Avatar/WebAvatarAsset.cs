// WebAvatarAsset.cs — canonical avatar data produced by importing a
// webavatar-rust NPZ file.  Float16 source arrays are widened to float32
// because Unity's GPU buffer types don't have a native half path on
// every backend.  Float32 also matches what the gsplat-unity renderer
// expects, so the runtime pass can be SetData-and-go.

using System;
using UnityEngine;

namespace WebAvatar
{
    public class WebAvatarAsset : ScriptableObject
    {
        [Serializable]
        public class MlpLayerData
        {
            public int InFeatures;
            public int OutFeatures;
            public int Activation; // 0 = none, 1 = relu-ish
            public int[] WeightShape;   // [Out, Partition, In] or [Out, In]
            public float[] Weights;
            public float[] Biases;
        }

        [Header("Canonical Gaussians (NPZ arrays)")]
        public int SplatCount;
        public Vector3[] Positions;        // _xyz  (SplatCount, 3)  f16 -> f32
        public Vector3[] Scales;           // _scaling
        public Vector4[] Rotations;        // _rotation  quaternion (w, x, y, z) — not normalised here
        public Color[] Sh0;                // _sh0  RGB DC term of SH band 0
        public float[] Opacities;          // opacity / 255  (0..1)

        [Header("Bounding box")]
        public Bounds Bounds;

        [Header("LBS (Linear Blend Skinning)")]
        public int JointCount;
        public Vector3[] Joints;           // t_joints (JointCount, 3) f16 -> f32
        public byte[] WeightsIdx;          // (SplatCount * 4) flat
        public byte[] WeightsVal;          // (SplatCount * 4) flat
        public const int WeightsPerSplat = 4;

        [Header("MLP (per-part feature generator)")]
        public int NumHeadParts;           // num_xyz_ft_head
        public MlpLayerData[] MlpLayers;   // layers.{0,2,4,6}.{weight,bias}

        [Header("Attribute (blend shapes & control points)")]
        public int PrunedSplatCount;       // length of rot_idxs / scale_idxs / color_idxs
        public int ControlPointCount;      // length of ctrl_feat_knn

        public ushort[] GsFeatKnn;         // (SplatCount,)
        public uint[] RotIdxs;             // (PrunedSplatCount,)
        public uint[] ScaleIdxs;
        public uint[] ColorIdxs;

        // Basis arrays are stored as flat float[PrunedSplatCount * Dim * 16] or
        // float[ControlPointCount * 3 * 16] for control points.  See Basis*Stride
        // helpers below for the per-element dimension.
        public float[] ColorBasis;         // (PrunedSplatCount, 3, 16)
        public float[] ScaleBasis;         // (PrunedSplatCount, 3, 16)
        public float[] RotationBasis;      // (PrunedSplatCount, 4, 16)

        public float[] ControlPointBasis;  // (ControlPointCount, 3, 16)
        public float[] ControlPointOffsets;// (ControlPointCount, 3) f32

        public ushort[] GsCtrlKnn;         // (SplatCount, 3) flat
        public float[] GsCtrlKnnWeights;   // (SplatCount, 3) flat
        public ushort[] CtrlFeatKnn;       // (ControlPointCount,)

        // Layout helpers — basis arrays are laid out as
        //   [splatIndex, basisDim, k]   in column-major with k-major inner loop.
        // 16 is the number of basis components per attribute dimension.
        public const int BasisComponents = 16;

        public static int BasisStride => BasisComponents;
        public static int RotationBasisStride => 4 * BasisComponents;
        public static int ControlPointBasisStride => 3 * BasisComponents;
        public static int SplatBasisStride => 3 * BasisComponents;

        public int ColorBasisSplatStride => SplatBasisStride;
        public int ScaleBasisSplatStride => SplatBasisStride;
        public int RotationBasisSplatStride => RotationBasisStride;
    }
}
