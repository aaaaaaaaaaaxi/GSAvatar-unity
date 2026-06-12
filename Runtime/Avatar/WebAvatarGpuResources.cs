// WebAvatarGpuResources.cs — owns all GPU buffers / compute shaders
// used by the per-frame pipeline, and exposes a single DispatchFrame
// call that drives the whole LBS → MLP → Attribute → Transform
// sequence.
//
// All buffers are GraphicsBuffer.Target.Structured (the downstream
// wu.yize.gsplat GraphicsBuffers are also Structured, which means we
// can pass them directly to a RWStructuredBuffer in the transform
// compute kernel — no extra copy step).
//
// The 4 per-frame scratch buffers (Positions, Rotations, LogScales,
// Sh0) are re-initialised to the canonical values at the start of
// every DispatchFrame call (see ResetScratchBuffers).  This mirrors
// the Rust pipeline's
//   encoder.copy_buffer_to_buffer(vertex_buffer_unmodified, ...)
// at the start of attribute.rs::compute: without the reset, LBS
// would read the previous frame's LBS-skinned position (compounds
// exponentially) and AttributeBasis would add basis deltas on top
// of the previous frame's already-delta'd value (compounds linearly),
// causing the avatar to drift chaotically after a few frames.

using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using WebAvatar.Pose;

namespace WebAvatar
{
    public class WebAvatarGpuResources : IDisposable
    {
        // ---- Counters (read-only) ----
        public readonly int SplatCount;
        public readonly int PrunedSplatCount;
        public readonly int ControlPointCount;
        public readonly int NumMlps;
        public readonly int NumHeadParts;
        public readonly int MlpOutFeatures;          // last layer out_features
        public readonly MlpLayerInfo[] Layers;
        public readonly int MaxMlpElements;

        // ---- Cached canonical data (re-uploaded each frame to reset
        //      the working scratch buffers; see ResetScratchBuffers). ----
        readonly WebAvatarAsset _asset;
        readonly Vector3[] _canonPos;
        readonly Vector4[] _canonRot;
        readonly Vector3[] _canonLogScale;
        readonly Vector3[] _canonSh0;

        // ---- Per-frame scratch buffers ----
        public GraphicsBuffer PositionsBuf;
        public GraphicsBuffer RotationsBuf;
        public GraphicsBuffer LogScalesBuf;
        public GraphicsBuffer Sh0Buf;
        public GraphicsBuffer OpacitiesBuf;
        public GraphicsBuffer CtrlPtDeltasBuf;

        // ---- MLP ping-pong ----
        public GraphicsBuffer MlpPingA;
        public GraphicsBuffer MlpPingB;
        public GraphicsBuffer MlpCoefsBuf;

        // ---- Static read-only buffers (uploaded once from the asset) ----
        public GraphicsBuffer WeightsValBuf;
        public GraphicsBuffer WeightsIdxBuf;
        public GraphicsBuffer GsFeatKnnBuf;
        public GraphicsBuffer PruneIndicesBuf;
        public GraphicsBuffer RotationBasisBuf;
        public GraphicsBuffer ScaleBasisBuf;
        public GraphicsBuffer ColorBasisBuf;
        public GraphicsBuffer MlpWeightsBuf;
        public GraphicsBuffer MlpBiasesBuf;
        public GraphicsBuffer CtrlFeatKnnBuf;
        public GraphicsBuffer CtrlPtBasisBuf;
        public GraphicsBuffer CtrlPtOffsetsBuf;
        public GraphicsBuffer GsCtrlKnnBuf;
        public GraphicsBuffer GsCtrlKnnWeightsBuf;
        public GraphicsBuffer JointMatsBuf;

        // ---- Pose uniform (cbuffer) ----
        public readonly Vector4[] PoseVec4 = new Vector4[20];
        public const int MlpPoseDim = 63;
        public const int MlpPoseBufferF32s = 80;
        public const int NumJoints = 55;

        // ---- Compute shaders ----
        readonly ComputeShader _lbsCs;
        readonly ComputeShader _mlpCs;
        readonly ComputeShader _attrBasisCs;
        readonly ComputeShader _attrDxyzCs;
        readonly ComputeShader _attrTransformCs;

        readonly int _kLbs;
        readonly int _kMlp;
        readonly int _kAttrBasis;
        readonly int _kAttrDxyz;
        readonly int _kAttrTransform;

        public struct MlpLayerInfo
        {
            public int InFeatures;
            public int OutFeatures;
            public int Activation;
            public int WeightOffset;   // float offset into MlpWeightsBuf
            public int WeightCount;
            public int BiasOffset;
            public int BiasCount;
        }

        bool _disposed;

        public WebAvatarGpuResources(WebAvatarAsset asset)
        {
            if (asset == null) throw new ArgumentNullException(nameof(asset));
            if (asset.SplatCount <= 0) throw new ArgumentException("Empty avatar");
            if (asset.MlpLayers == null || asset.MlpLayers.Length == 0)
                throw new ArgumentException("Avatar has no MLP layers");

            SplatCount = asset.SplatCount;
            PrunedSplatCount = asset.PrunedSplatCount;
            ControlPointCount = asset.ControlPointCount;
            NumHeadParts = asset.NumHeadParts;
            MlpOutFeatures = asset.MlpLayers[asset.MlpLayers.Length - 1].OutFeatures;

            int numMlps = 0;
            int weightCursor = 0, biasCursor = 0;
            int maxElems = 0;
            Layers = new MlpLayerInfo[asset.MlpLayers.Length];
            for (int i = 0; i < asset.MlpLayers.Length; i++)
            {
                var L = asset.MlpLayers[i];
                if (L.WeightShape.Length == 3) numMlps = L.WeightShape[1];
                int wCount = L.Weights.Length;
                int bCount = L.Biases.Length;
                Layers[i] = new MlpLayerInfo
                {
                    InFeatures  = L.InFeatures,
                    OutFeatures = L.OutFeatures,
                    Activation  = L.Activation,
                    WeightOffset = weightCursor,
                    WeightCount  = wCount,
                    BiasOffset   = biasCursor,
                    BiasCount    = bCount,
                };
                weightCursor += wCount;
                biasCursor  += bCount;
                maxElems = Mathf.Max(maxElems,
                    Mathf.Max(numMlps * L.OutFeatures, numMlps * L.InFeatures));
            }
            NumMlps = numMlps;
            MaxMlpElements = Mathf.Max(1, maxElems);

            // Compute shader assets
            _lbsCs          = LoadCs("LbsSkinning");
            _mlpCs          = LoadCs("MlpForward");
            _attrBasisCs    = LoadCs("AttributeBasis");
            _attrDxyzCs     = LoadCs("AttributeDxyz");
            _attrTransformCs= LoadCs("AttributeTransform");

            _kLbs           = _lbsCs.FindKernel("LbsSkinning");
            _kMlp           = _mlpCs.FindKernel("MlpForward");
            _kAttrBasis     = _attrBasisCs.FindKernel("AttributeBasis");
            _kAttrDxyz      = _attrDxyzCs.FindKernel("AttributeDxyz");
            _kAttrTransform = _attrTransformCs.FindKernel("AttributeTransform");

            // Allocate buffers
            PositionsBuf    = AllocStructured(SplatCount, sizeof(float) * 3);
            RotationsBuf    = AllocStructured(SplatCount, sizeof(float) * 4);
            LogScalesBuf    = AllocStructured(SplatCount, sizeof(float) * 3);
            Sh0Buf          = AllocStructured(SplatCount, sizeof(float) * 3);
            OpacitiesBuf    = AllocStructured(SplatCount, sizeof(float));
            CtrlPtDeltasBuf = AllocStructured(Mathf.Max(1, ControlPointCount), sizeof(float) * 3);

            MlpPingA        = AllocStructured(MaxMlpElements, sizeof(float));
            MlpPingB        = AllocStructured(MaxMlpElements, sizeof(float));
            MlpCoefsBuf     = AllocStructured(NumMlps * MlpOutFeatures, sizeof(float));

            WeightsValBuf       = AllocStructured(SplatCount, sizeof(uint));
            WeightsIdxBuf       = AllocStructured(SplatCount, sizeof(uint));
            GsFeatKnnBuf        = AllocStructured(SplatCount, sizeof(uint));
            PruneIndicesBuf     = AllocStructured(3 * Mathf.Max(1, PrunedSplatCount), sizeof(uint));
            RotationBasisBuf    = AllocStructured(Mathf.Max(1, PrunedSplatCount) * 4 * 16, sizeof(float));
            ScaleBasisBuf       = AllocStructured(Mathf.Max(1, PrunedSplatCount) * 3 * 16, sizeof(float));
            ColorBasisBuf       = AllocStructured(Mathf.Max(1, PrunedSplatCount) * 3 * 16, sizeof(float));
            MlpWeightsBuf       = AllocStructured(Mathf.Max(1, weightCursor), sizeof(float));
            MlpBiasesBuf        = AllocStructured(Mathf.Max(1, biasCursor),  sizeof(float));
            CtrlFeatKnnBuf      = AllocStructured(Mathf.Max(1, ControlPointCount), sizeof(uint));
            CtrlPtBasisBuf      = AllocStructured(Mathf.Max(1, ControlPointCount) * 3 * 16, sizeof(float));
            CtrlPtOffsetsBuf    = AllocStructured(Mathf.Max(1, ControlPointCount), sizeof(float) * 3);
            GsCtrlKnnBuf        = AllocStructured(SplatCount * 3, sizeof(uint));
            GsCtrlKnnWeightsBuf = AllocStructured(SplatCount * 3, sizeof(float));
            JointMatsBuf        = AllocStructured(NumJoints * 4, sizeof(float) * 4);

            UploadStaticData(asset);
            // Cache the canonical per-splat arrays in CPU memory so the
            // per-frame ResetScratchBuffers() can re-upload them without
            // re-fetching from the ScriptableObject each time.
            _asset         = asset;
            _canonPos      = (Vector3[])asset.Positions.Clone();
            _canonRot      = (Vector4[])asset.Rotations.Clone();
            _canonLogScale = (Vector3[])asset.Scales.Clone();
            _canonSh0      = new Vector3[SplatCount];
            for (int i = 0; i < SplatCount; i++)
                _canonSh0[i] = new Vector3(asset.Sh0[i].r, asset.Sh0[i].g, asset.Sh0[i].b);
            UploadScratchInitial();
        }

        ComputeShader LoadCs(string name)
        {
            var cs = Resources.Load<ComputeShader>(name);
            if (cs == null) throw new InvalidOperationException(
                $"Compute shader '{name}' not found in any Resources folder " +
                "of the WebAvatar package");
            return cs;
        }

        static GraphicsBuffer AllocStructured(int count, int stride)
        {
            count = Mathf.Max(1, count);
            return new GraphicsBuffer(GraphicsBuffer.Target.Structured, count, stride);
        }

        void UploadStaticData(WebAvatarAsset asset)
        {
            // ---- LBS weights: pack 4x u8 into u32 (low byte = w0) ----
            var wv = new uint[SplatCount];
            var wi = new uint[SplatCount];
            for (int i = 0; i < SplatCount; i++)
            {
                int o = i * 4;
                wv[i] = (uint)asset.WeightsVal[o + 0]
                      | ((uint)asset.WeightsVal[o + 1] << 8)
                      | ((uint)asset.WeightsVal[o + 2] << 16)
                      | ((uint)asset.WeightsVal[o + 3] << 24);
                wi[i] = (uint)asset.WeightsIdx[o + 0]
                      | ((uint)asset.WeightsIdx[o + 1] << 8)
                      | ((uint)asset.WeightsIdx[o + 2] << 16)
                      | ((uint)asset.WeightsIdx[o + 3] << 24);
            }
            WeightsValBuf.SetData(wv);
            WeightsIdxBuf.SetData(wi);

            // ---- GsFeatKnn as u32 (low 16 bits = u16) ----
            if (asset.GsFeatKnn != null && asset.GsFeatKnn.Length == SplatCount)
            {
                var tmp = new uint[SplatCount];
                for (int i = 0; i < SplatCount; i++) tmp[i] = asset.GsFeatKnn[i];
                GsFeatKnnBuf.SetData(tmp);
            }
            else
            {
                GsFeatKnnBuf.SetData(new uint[SplatCount]);
            }

            // ---- Prune indices: [rot | scale | color] concat ----
            if (PrunedSplatCount > 0)
            {
                var tmp = new uint[3 * PrunedSplatCount];
                for (int i = 0; i < PrunedSplatCount; i++) tmp[i] = asset.RotIdxs[i];
                for (int i = 0; i < PrunedSplatCount; i++)
                    tmp[PrunedSplatCount + i] = asset.ScaleIdxs[i];
                for (int i = 0; i < PrunedSplatCount; i++)
                    tmp[2 * PrunedSplatCount + i] = asset.ColorIdxs[i];
                PruneIndicesBuf.SetData(tmp);

                if (asset.RotationBasis != null && asset.RotationBasis.Length > 0)
                    RotationBasisBuf.SetData(asset.RotationBasis);
                if (asset.ScaleBasis != null && asset.ScaleBasis.Length > 0)
                    ScaleBasisBuf.SetData(asset.ScaleBasis);
                if (asset.ColorBasis != null && asset.ColorBasis.Length > 0)
                    ColorBasisBuf.SetData(asset.ColorBasis);
            }

            // ---- MLP weights / biases, re-permuted to [num_mlps, Out, In] ----
            int totalWeights = 0, totalBiases = 0;
            for (int i = 0; i < asset.MlpLayers.Length; i++)
            {
                totalWeights += asset.MlpLayers[i].Weights.Length;
                totalBiases  += asset.MlpLayers[i].Biases.Length;
            }
            var wAll = new float[totalWeights];
            var bAll = new float[totalBiases];
            int wOff = 0, bOff = 0;
            for (int li = 0; li < asset.MlpLayers.Length; li++)
            {
                var L = asset.MlpLayers[li];
                if (L.WeightShape.Length == 3)
                {
                    int outD = L.WeightShape[0];
                    int mlpD = L.WeightShape[1];
                    int inD  = L.WeightShape[2];
                    var src = L.Weights;
                    for (int o = 0; o < outD; o++)
                    for (int m = 0; m < mlpD; m++)
                    for (int ii = 0; ii < inD; ii++)
                    {
                        int srcIdx = (o * mlpD + m) * inD + ii;
                        int dstIdx = (m * outD + o) * inD + ii;
                        wAll[wOff + dstIdx] = src[srcIdx];
                    }
                    wOff += src.Length;

                    var bSrc = L.Biases;
                    for (int o = 0; o < outD; o++)
                    for (int m = 0; m < mlpD; m++)
                    {
                        bAll[bOff + m * outD + o] = bSrc[o * mlpD + m];
                    }
                    bOff += bSrc.Length;
                }
                else
                {
                    Array.Copy(L.Weights, 0, wAll, wOff, L.Weights.Length);
                    wOff += L.Weights.Length;
                    Array.Copy(L.Biases, 0, bAll, bOff, L.Biases.Length);
                    bOff += L.Biases.Length;
                }
            }
            MlpWeightsBuf.SetData(wAll);
            MlpBiasesBuf.SetData(bAll);

            // ---- Control point KNN, basis, offsets ----
            if (ControlPointCount > 0)
            {
                if (asset.CtrlFeatKnn != null && asset.CtrlFeatKnn.Length == ControlPointCount)
                {
                    var tmp = new uint[ControlPointCount];
                    for (int i = 0; i < ControlPointCount; i++) tmp[i] = asset.CtrlFeatKnn[i];
                    CtrlFeatKnnBuf.SetData(tmp);
                }
                if (asset.ControlPointBasis != null && asset.ControlPointBasis.Length > 0)
                    CtrlPtBasisBuf.SetData(asset.ControlPointBasis);
                if (asset.ControlPointOffsets != null && asset.ControlPointOffsets.Length == ControlPointCount * 3)
                {
                    var tmp = new Vector3[ControlPointCount];
                    for (int i = 0; i < ControlPointCount; i++)
                        tmp[i] = new Vector3(
                            asset.ControlPointOffsets[3 * i + 0],
                            asset.ControlPointOffsets[3 * i + 1],
                            asset.ControlPointOffsets[3 * i + 2]);
                    CtrlPtOffsetsBuf.SetData(tmp);
                }
            }

            // ---- GsCtrlKnn / weights ----
            if (asset.GsCtrlKnn != null && asset.GsCtrlKnn.Length == SplatCount * 3)
            {
                var tmp = new uint[SplatCount * 3];
                for (int i = 0; i < SplatCount * 3; i++) tmp[i] = asset.GsCtrlKnn[i];
                GsCtrlKnnBuf.SetData(tmp);
            }
            if (asset.GsCtrlKnnWeights != null && asset.GsCtrlKnnWeights.Length == SplatCount * 3)
                GsCtrlKnnWeightsBuf.SetData(asset.GsCtrlKnnWeights);
        }

        void UploadScratchInitial()
        {
            // Initialise the per-frame scratch buffers to the canonical values.
            // After construction this is one-shot; per-frame re-initialisation
            // is handled by ResetScratchBuffers() called at the top of every
            // DispatchFrame().
            PositionsBuf.SetData(_canonPos);
            RotationsBuf.SetData(_canonRot);
            LogScalesBuf.SetData(_canonLogScale);
            Sh0Buf.SetData(_canonSh0);

            // Opacities are not touched by any per-frame pass so the canonical
            // upload is one-and-done.
            var opa = new float[SplatCount];
            for (int i = 0; i < SplatCount; i++) opa[i] = _asset.Opacities[i];
            OpacitiesBuf.SetData(opa);
        }

        /// <summary>
        /// Reset the per-frame scratch buffers to the canonical (unmodified)
        /// Gaussian state.  Mirrors the Rust pipeline's
        ///   encoder.copy_buffer_to_buffer(&pc.vertex_buffer_unmodified, ...)
        /// call at the start of attribute.rs::compute, which is what stops
        /// the per-frame state from compounding.
        ///
        /// Without this reset the C# pipeline has two compounding bugs:
        ///   * LBS reads <c>PositionsBuf</c>/<c>RotationsBuf</c> as input
        ///     and writes back, so the LBS would otherwise be applied on
        ///     top of last frame's already-skinned state (drift grows
        ///     geometrically).
        ///   * AttributeBasis ADDs the per-frame basis delta to
        ///     <c>RotationsBuf</c>/<c>LogScalesBuf</c>/<c>Sh0Buf</c>, so
        ///     without a reset the deltas accumulate linearly each frame.
        ///
        /// The reset only touches the four per-frame scratch buffers;
        /// the opacity buffer is left at its construction-time value
        /// because no per-frame pass writes to it.
        /// </summary>
        void ResetScratchBuffers()
        {
            PositionsBuf.SetData(_canonPos);
            RotationsBuf.SetData(_canonRot);
            LogScalesBuf.SetData(_canonLogScale);
            Sh0Buf.SetData(_canonSh0);
        }

        // ==================================================================
        // Per-frame dispatch
        // ==================================================================

        /// <summary>
        /// Run one full frame of the pipeline.
        /// <paramref name="jointMatsFlat"/> is 55 * 16 floats in Unity's
        /// Matrix4x4 column-major storage (= the bytes that
        /// SetData(Matrix4x4[]) would produce).
        /// </summary>
        public void DispatchFrame(
            PoseFrame pose,
            float[] jointMatsFlat,
            Matrix4x4 globalMat4x4,
            GraphicsBuffer outPos,
            GraphicsBuffer outScale,
            GraphicsBuffer outRot,
            GraphicsBuffer outColor)
        {
            // Reset the working scratch buffers to their canonical (un-LBS'd,
            // un-delta'd) state BEFORE LBS runs.  This is the per-frame
            // equivalent of Rust's copy_buffer_to_buffer call at the top of
            // attribute.rs::compute, and is what stops the LBS / attribute
            // state from compounding frame over frame.  See ResetScratchBuffers
            // for the full reasoning.
            ResetScratchBuffers();

            FillPoseVec4(pose);
            JointMatsBuf.SetData(jointMatsFlat);

            // ---- LBS ----
            _lbsCs.SetInt("_SplatCount", SplatCount);
            _lbsCs.SetVector("_GlobalMat0", globalMat4x4.GetColumn(0));
            _lbsCs.SetVector("_GlobalMat1", globalMat4x4.GetColumn(1));
            _lbsCs.SetVector("_GlobalMat2", globalMat4x4.GetColumn(2));
            _lbsCs.SetVector("_GlobalMat3", globalMat4x4.GetColumn(3));
            _lbsCs.SetBuffer(_kLbs, "_JointMats",  JointMatsBuf);
            _lbsCs.SetBuffer(_kLbs, "InPositions", PositionsBuf);
            _lbsCs.SetBuffer(_kLbs, "InRotations", RotationsBuf);
            _lbsCs.SetBuffer(_kLbs, "WeightsVal",  WeightsValBuf);
            _lbsCs.SetBuffer(_kLbs, "WeightsIdx",  WeightsIdxBuf);
            _lbsCs.SetBuffer(_kLbs, "OutPositions", PositionsBuf);
            _lbsCs.SetBuffer(_kLbs, "OutRotations", RotationsBuf);
            _lbsCs.Dispatch(_kLbs, (SplatCount + 255) / 256, 1, 1);

            // ---- MLP forward (ping-pong A <-> B, last layer writes to MlpCoefsBuf) ----
            GraphicsBuffer inBuf = MlpPingA;
            for (int li = 0; li < Layers.Length; li++)
            {
                var L = Layers[li];
                bool isLast = li == Layers.Length - 1;
                GraphicsBuffer outBuf = isLast
                    ? MlpCoefsBuf
                    : (li % 2 == 0 ? MlpPingB : MlpPingA);

                _mlpCs.SetInt("InFeatures",  L.InFeatures);
                _mlpCs.SetInt("OutFeatures", L.OutFeatures);
                _mlpCs.SetInt("Activation",  L.Activation);
                _mlpCs.SetInt("NumMlps",     NumMlps);
                _mlpCs.SetInt("NumHeadParts",NumHeadParts);
                _mlpCs.SetInt("NumPoseDim",  MlpPoseDim);
                _mlpCs.SetInt("UsePoseUniform", li == 0 ? 1 : 0);
                _mlpCs.SetVectorArray("_Pose", PoseVec4);
                _mlpCs.SetBuffer(_kMlp, "InW",  MlpWeightsBuf);
                _mlpCs.SetBuffer(_kMlp, "InB",  MlpBiasesBuf);
                _mlpCs.SetBuffer(_kMlp, "InX",  inBuf);
                _mlpCs.SetBuffer(_kMlp, "OutY", outBuf);
                // The weight / bias sub-range is encoded via an int offset
                // uniform on the shader, but ComputeShader.SetBuffer does
                // not accept a per-call range; we instead pass the offsets
                // as uniforms and the shader reads from `InW[offset + k]`.
                // (See mlp.wgsl — same shape.)
                _mlpCs.SetInt("_WeightOffset", L.WeightOffset);
                _mlpCs.SetInt("_BiasOffset",   L.BiasOffset);
                _mlpCs.Dispatch(_kMlp,
                    (L.OutFeatures + 63) / 64, NumMlps, 1);

                inBuf = outBuf;
            }

            // ---- Attribute basis ----
            if (PrunedSplatCount > 0)
            {
                _attrBasisCs.SetInt("PrunedSplatCount", PrunedSplatCount);
                _attrBasisCs.SetInt("CoefStride",       MlpOutFeatures);
                _attrBasisCs.SetBuffer(_kAttrBasis, "GsFeatKnn",      GsFeatKnnBuf);
                _attrBasisCs.SetBuffer(_kAttrBasis, "PruneIndices",   PruneIndicesBuf);
                _attrBasisCs.SetBuffer(_kAttrBasis, "RotationBasis",  RotationBasisBuf);
                _attrBasisCs.SetBuffer(_kAttrBasis, "ScaleBasis",     ScaleBasisBuf);
                _attrBasisCs.SetBuffer(_kAttrBasis, "ColorBasis",     ColorBasisBuf);
                _attrBasisCs.SetBuffer(_kAttrBasis, "Coefs",          MlpCoefsBuf);
                _attrBasisCs.SetBuffer(_kAttrBasis, "OutRotations",   RotationsBuf);
                _attrBasisCs.SetBuffer(_kAttrBasis, "OutLogScales",   LogScalesBuf);
                _attrBasisCs.SetBuffer(_kAttrBasis, "OutSh0",         Sh0Buf);
                _attrBasisCs.Dispatch(_kAttrBasis,
                    (PrunedSplatCount + 255) / 256, 1, 1);
            }

            // ---- Attribute dxyz ----
            if (ControlPointCount > 0)
            {
                _attrDxyzCs.SetInt("ControlPointCount", ControlPointCount);
                _attrDxyzCs.SetInt("CoefStride",        MlpOutFeatures);
                _attrDxyzCs.SetBuffer(_kAttrDxyz, "CtrlFeatKnn",   CtrlFeatKnnBuf);
                _attrDxyzCs.SetBuffer(_kAttrDxyz, "CtrlPtBasis",   CtrlPtBasisBuf);
                _attrDxyzCs.SetBuffer(_kAttrDxyz, "CtrlPtOffsets", CtrlPtOffsetsBuf);
                _attrDxyzCs.SetBuffer(_kAttrDxyz, "Coefs",         MlpCoefsBuf);
                _attrDxyzCs.SetBuffer(_kAttrDxyz, "OutCtrlPtDeltas", CtrlPtDeltasBuf);
                _attrDxyzCs.Dispatch(_kAttrDxyz,
                    (ControlPointCount + 255) / 256, 1, 1);
            }

            // ---- Attribute transform (writes to gsplat buffers) ----
            _attrTransformCs.SetInt("SplatCount", SplatCount);
            _attrTransformCs.SetBuffer(_kAttrTransform, "InPositions",        PositionsBuf);
            _attrTransformCs.SetBuffer(_kAttrTransform, "InLogScales",        LogScalesBuf);
            _attrTransformCs.SetBuffer(_kAttrTransform, "InRotations",        RotationsBuf);
            _attrTransformCs.SetBuffer(_kAttrTransform, "InSh0",              Sh0Buf);
            _attrTransformCs.SetBuffer(_kAttrTransform, "InOpacities",        OpacitiesBuf);
            _attrTransformCs.SetBuffer(_kAttrTransform, "GsCtrlKnn",          GsCtrlKnnBuf);
            _attrTransformCs.SetBuffer(_kAttrTransform, "GsCtrlKnnWeights",   GsCtrlKnnWeightsBuf);
            _attrTransformCs.SetBuffer(_kAttrTransform, "CtrlPtDeltas",       CtrlPtDeltasBuf);
            _attrTransformCs.SetBuffer(_kAttrTransform, "OutPositions",       outPos);
            _attrTransformCs.SetBuffer(_kAttrTransform, "OutScales",          outScale);
            _attrTransformCs.SetBuffer(_kAttrTransform, "OutRotations",       outRot);
            _attrTransformCs.SetBuffer(_kAttrTransform, "OutColors",          outColor);
            _attrTransformCs.Dispatch(_kAttrTransform,
                (SplatCount + 255) / 256, 1, 1);
        }

        // ---- pose buffer ----

        void FillPoseVec4(PoseFrame pose)
        {
            for (int i = 0; i < 20; i++) PoseVec4[i] = Vector4.zero;

            // pose: 63 floats (joints 1..21 axis-angle), padded to 64 with 0
            if (pose.pose != null)
            {
                int n = Mathf.Min(MlpPoseDim, pose.pose.Length);
                for (int k = 0; k < n; k++)
                    PoseVec4[k / 4][k % 4] = pose.pose[k];
            }

            // expression: 10 floats padded to 16, starts at float index 64
            if (pose.expression != null && pose.expression.Length > 0)
            {
                int n = Mathf.Min(10, pose.expression.Length);
                int exprVecStart = 16; // vector index 16 = float index 64
                for (int k = 0; k < n; k++)
                    PoseVec4[exprVecStart + k / 4][k % 4] = pose.expression[k];
            }
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            DisposeBuf(ref PositionsBuf);
            DisposeBuf(ref RotationsBuf);
            DisposeBuf(ref LogScalesBuf);
            DisposeBuf(ref Sh0Buf);
            DisposeBuf(ref OpacitiesBuf);
            DisposeBuf(ref CtrlPtDeltasBuf);
            DisposeBuf(ref MlpPingA);
            DisposeBuf(ref MlpPingB);
            DisposeBuf(ref MlpCoefsBuf);
            DisposeBuf(ref WeightsValBuf);
            DisposeBuf(ref WeightsIdxBuf);
            DisposeBuf(ref GsFeatKnnBuf);
            DisposeBuf(ref PruneIndicesBuf);
            DisposeBuf(ref RotationBasisBuf);
            DisposeBuf(ref ScaleBasisBuf);
            DisposeBuf(ref ColorBasisBuf);
            DisposeBuf(ref MlpWeightsBuf);
            DisposeBuf(ref MlpBiasesBuf);
            DisposeBuf(ref CtrlFeatKnnBuf);
            DisposeBuf(ref CtrlPtBasisBuf);
            DisposeBuf(ref CtrlPtOffsetsBuf);
            DisposeBuf(ref GsCtrlKnnBuf);
            DisposeBuf(ref GsCtrlKnnWeightsBuf);
            DisposeBuf(ref JointMatsBuf);
        }

        static void DisposeBuf(ref GraphicsBuffer b)
        {
            b?.Dispose();
            b = null;
        }
    }
}
