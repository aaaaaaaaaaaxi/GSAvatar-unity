// WebAvatarSkinning.cs — CPU-side SMPL-X skinning math.
//
// Mirrors the per-frame work that the Rust pipeline does in
// e:\HKUSTGZ\HoloSoul\webavatar-rust\src\lbs.rs:
//
//   rigid_transform(joint_rotations, joint_positions) -> A_pose
//   A_canonical_inv            = A_bigpose.inverse()       (precomputed once)
//   G_i                        = A_pose[i] * A_canonical_inv[i]
//
// The output is the per-joint skinning matrix G.  The compute shader
// blends the four G matrices weighted by u8 weights, then applies
// the global rigid transform on top.
//
// All math is in Unity's column-major Matrix4x4 world.  HLSL's
// `float4x4` row-major is reconciled by Unity's SetMatrixArray.

using UnityEngine;

namespace WebAvatar
{
    public static class WebAvatarSkinning
    {
        public const int SMPL_NUM_JOINTS = 55;
        public const int SMPL_POSE_DIM = SMPL_NUM_JOINTS * 3; // 165

        /// <summary>SMPL-X kinematic tree: parent index per joint, -1 == root.</summary>
        public static readonly int[] PARENT = new int[SMPL_NUM_JOINTS]
        {
            -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9,
            12, 13, 14, 16, 17, 18, 19, 15, 15, 15, 20,
            25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38,
            21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53,
        };

        // ------------------------------------------------------------------
        // Canonical (A-pose) precompute
        // ------------------------------------------------------------------

        /// <summary>
        /// Build the "big pose" axis-angle vector with hips at ±25° so the
        /// canonical A-pose is the rest configuration.  See lbs.rs:135.
        /// </summary>
        public static void InitSmplPose(float[] outPose)
        {
            if (outPose == null || outPose.Length != SMPL_POSE_DIM)
                throw new System.ArgumentException(
                    $"pose buffer must be length {SMPL_POSE_DIM}");
            for (int i = 0; i < SMPL_POSE_DIM; i++) outPose[i] = 0f;
            // joint 5 axis-angle = (+25°, 0, 0);  joint 8 = (-25°, 0, 0)
            outPose[5] =  25f * Mathf.Deg2Rad;
            outPose[8] = -25f * Mathf.Deg2Rad;
        }

        /// <summary>
        /// Precompute A_canonical_inv once.  Store the result for use in
        /// every later <see cref="ComputePoseJointMats"/> call.
        /// </summary>
        public static void ComputeCanonicalJointMatsInv(
            Vector3[] joints, Matrix4x4[] outAInv)
        {
            var bigpose = new float[SMPL_POSE_DIM];
            InitSmplPose(bigpose);
            var bigRotations = new Matrix4x4[SMPL_NUM_JOINTS];
            PoseToJointRotations(bigpose, bigRotations);
            var bigMats = new Matrix4x4[SMPL_NUM_JOINTS];
            RigidTransform(bigRotations, joints, bigMats);
            for (int i = 0; i < SMPL_NUM_JOINTS; i++)
                outAInv[i] = bigMats[i].inverse;
        }

        // ------------------------------------------------------------------
        // Per-frame
        // ------------------------------------------------------------------

        /// <summary>
        /// Compute per-joint skinning matrices G_i = A_pose[i] * A_canonical_inv[i]
        /// for the given pose.  <paramref name="aCanonicalInv"/> is the
        /// precomputed output of <see cref="ComputeCanonicalJointMatsInv"/>.
        /// </summary>
        public static void ComputePoseJointMats(
            float[] pose, Vector3[] joints, Matrix4x4[] aCanonicalInv,
            Matrix4x4[] outG)
        {
            var rotations = new Matrix4x4[SMPL_NUM_JOINTS];
            PoseToJointRotations(pose, rotations);
            var aPose = new Matrix4x4[SMPL_NUM_JOINTS];
            RigidTransform(rotations, joints, aPose);
            for (int i = 0; i < SMPL_NUM_JOINTS; i++)
                outG[i] = aPose[i] * aCanonicalInv[i];
        }

        /// <summary>
        /// Build the global rigid transform (rotation 3x3 row-major + translation)
        /// that is applied on top of the LBS output.  Layout matches lbs.rs:253-258.
        ///
        /// The Rust reference applies `M_rust * v = R * v + T` where R is the
        /// row-major rotation read straight from the JSON's "Rh" field
        /// (each inner array is one row of R).  cgmath stores the transposed
        /// matrix in `rotation`, then `Matrix4::from_cols(rotation.x, ...)`
        /// makes the matrix columns equal to rotation's basis vectors
        /// (which are R's rows).  Reproducing that on the Unity side
        /// means: column `c` of the Matrix4x4 = row `c` of R, i.e. the
        /// `c`-th triple of the input buffer.  Equivalently we lay R out
        /// into the matrix row by row, and let Unity's column-major
        /// storage pick the columns up automatically.
        /// </summary>
        public static Matrix4x4 BuildGlobalMatrix(
            Vector3 transl, float[] rotationRowMajor)
        {
            if (rotationRowMajor == null || rotationRowMajor.Length != 9)
                throw new System.ArgumentException("rotation must be length 9");
            var m = new Matrix4x4();
            // m.mRC == element at row R, column C.  We want R as a
            // standard row-major 3x3: row 0 = (r[0], r[1], r[2]), etc.
            // The translation lives in column 3.
            m.m00 = rotationRowMajor[0]; m.m01 = rotationRowMajor[1]; m.m02 = rotationRowMajor[2];
            m.m10 = rotationRowMajor[3]; m.m11 = rotationRowMajor[4]; m.m12 = rotationRowMajor[5];
            m.m20 = rotationRowMajor[6]; m.m21 = rotationRowMajor[7]; m.m22 = rotationRowMajor[8];
            m.m03 = transl.x; m.m13 = transl.y; m.m23 = transl.z;
            m.m30 = 0f;       m.m31 = 0f;       m.m32 = 0f;       m.m33 = 1f;
            return m;
        }

        // ------------------------------------------------------------------
        // Internals
        // ------------------------------------------------------------------

        /// <summary>
        /// Convert 165 axis-angle floats (3 per joint) into 55 rotation
        /// matrices.  Mirrors lbs.rs pose_to_matrix().
        /// </summary>
        static void PoseToJointRotations(float[] pose, Matrix4x4[] outRotations)
        {
            for (int j = 0; j < SMPL_NUM_JOINTS; j++)
            {
                int o = j * 3;
                float rx = pose[o + 0];
                float ry = pose[o + 1];
                float rz = pose[o + 2];
                var v = new Vector3(rx, ry, rz);
                float angle = v.magnitude;
                if (angle < 1e-7f)
                {
                    outRotations[j] = Matrix4x4.identity;
                }
                else
                {
                    var axis = v / angle;
                    var q = Quaternion.AngleAxis(angle * Mathf.Rad2Deg, axis);
                    outRotations[j] = Matrix4x4.Rotate(q);
                }
            }
        }

        /// <summary>
        /// Accumulate joint local transforms along the kinematic tree.
        /// Mirrors lbs.rs rigid_transform().  joint_positions are the
        /// canonical t_joints; the root's offset is its canonical
        /// position (it has no parent), and every other joint's offset
        /// is t_i - t_parent.
        /// </summary>
        static void RigidTransform(
            Matrix4x4[] jointRotations, Vector3[] jointPositions,
            Matrix4x4[] outJointMats)
        {
            // Build per-joint local matrices: [R | offset].  For i >= 1
            // the offset is the canonical position relative to the
            // parent.  The root (PARENT[0] == -1) has no parent, so its
            // offset is just its own canonical position — the same
            // convention used in lbs.rs:84.  Leaving it at Vector3.zero
            // makes A_canonical_inv[0] lose the root's translation,
            // which then bleeds into G_0 and the LBS rotates vertices
            // around the origin instead of around the root's position;
            // limbs (which carry non-trivial root weight through the
            // spine → shoulder/hip chain) end up shifted by
            // w_root * (I - R_G_root) * t_joints[0], distorting them
            // relative to the body while the head (dominated by joints
            // 12..14) is unaffected.
            var offsets = new Vector3[SMPL_NUM_JOINTS];
            offsets[0] = jointPositions[0];
            for (int i = 1; i < SMPL_NUM_JOINTS; i++)
            {
                int p = PARENT[i];
                offsets[i] = jointPositions[i] - jointPositions[p];
            }
            for (int i = 0; i < SMPL_NUM_JOINTS; i++)
            {
                var m = jointRotations[i];
                // Place translation in the 4th column, keep w-component 1
                m.m03 = offsets[i].x;
                m.m13 = offsets[i].y;
                m.m23 = offsets[i].z;
                m.m33 = 1f;
                outJointMats[i] = m;
            }
            // Accumulate along the kinematic chain
            for (int i = 1; i < SMPL_NUM_JOINTS; i++)
            {
                int p = PARENT[i];
                outJointMats[i] = outJointMats[p] * outJointMats[i];
            }
        }
    }
}
