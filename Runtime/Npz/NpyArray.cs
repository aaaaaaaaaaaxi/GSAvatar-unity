// NpyArray.cs — typed views over a raw NPY payload.
//
// The payload is just NumPy's raw buffer (C-order, native dtype).  We keep
// it as a byte[] and expose typed read-only spans / arrays so importers
// can pick the precision they need without copying more than once.

using System;
using System.Buffers.Binary;

namespace WebAvatar.Npz
{
    public sealed class NpyArray
    {
        public NpyHeader Header { get; }
        public byte[] Data { get; }
        public string Name { get; }

        public NpyArray(string name, NpyHeader header, byte[] data)
        {
            if (data.Length != header.DataSize)
                throw new ArgumentException(
                    $"NPY payload size {data.Length} != header {header.DataSize} for '{name}'");
            Name = name;
            Header = header;
            Data = data;
        }

        // ------------------------------------------------------------------
        // Typed views
        // ------------------------------------------------------------------

        public ReadOnlySpan<byte> AsUInt8()
        {
            if (Header.DType != NpyDType.UInt8) throw new InvalidOperationException("not u1: " + Name);
            return Data;
        }

        public ushort[] ToUInt16()
        {
            if (Header.DType != NpyDType.UInt16) throw new InvalidOperationException("not u2: " + Name);
            var n = Header.NumElements;
            var result = new ushort[n];
            if (Header.IsLittleEndian)
            {
                Buffer.BlockCopy(Data, 0, result, 0, n * 2);
            }
            else
            {
                for (int i = 0; i < n; i++)
                    result[i] = BinaryPrimitives.ReadUInt16BigEndian(Data.AsSpan(i * 2, 2));
            }
            return result;
        }

        public uint[] ToUInt32()
        {
            if (Header.DType != NpyDType.UInt32) throw new InvalidOperationException("not u4: " + Name);
            var n = Header.NumElements;
            var result = new uint[n];
            if (Header.IsLittleEndian)
            {
                Buffer.BlockCopy(Data, 0, result, 0, n * 4);
            }
            else
            {
                for (int i = 0; i < n; i++)
                    result[i] = BinaryPrimitives.ReadUInt32BigEndian(Data.AsSpan(i * 4, 4));
            }
            return result;
        }

        // f16 -> f32, on the fly.  Unity has no native half-precision type, so
        // importers almost always want Float32.  We re-use Half helper from
        // System.Buffers.Binary or convert manually via the IEEE-754 layout.
        public float[] ToFloat32()
        {
            if (Header.DType == NpyDType.Float32)
            {
                var n = Header.NumElements;
                var result = new float[n];
                if (Header.IsLittleEndian)
                {
                    Buffer.BlockCopy(Data, 0, result, 0, n * 4);
                }
                else
                {
                    for (int i = 0; i < n; i++)
                    {
                        int bits = BinaryPrimitives.ReadInt32BigEndian(Data.AsSpan(i * 4, 4));
                        result[i] = BitConverter.Int32BitsToSingle(bits);
                    }
                }
                return result;
            }
            if (Header.DType == NpyDType.Float16)
            {
                var n = Header.NumElements;
                var result = new float[n];
                for (int i = 0; i < n; i++)
                {
                    ushort bits = BinaryPrimitives.ReadUInt16LittleEndian(Data.AsSpan(i * 2, 2));
                    result[i] = HalfToFloat(bits);
                }
                return result;
            }
            throw new InvalidOperationException(
                $"Cannot view '{Name}' (dtype={Header.DType}) as float32");
        }

        // Half-precision -> single-precision.  Implements the IEEE-754 binary16
        // format bit-for-bit so we don't pull in the System.Half dependency
        // (available only in .NET 5+ / Unity 2022.2+ on a per-platform basis).
        static float HalfToFloat(ushort bits)
        {
            uint sign = (uint)(bits >> 15) & 0x1u;
            uint exp  = (uint)(bits >> 10) & 0x1Fu;
            uint mant = (uint)bits         & 0x3FFu;

            uint f32Bits;
            if (exp == 0)
            {
                if (mant == 0)
                {
                    f32Bits = sign << 31;
                }
                else
                {
                    // subnormal half -> normalized single
                    while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
                    exp++; mant &= 0x3FFu;
                    uint fExp = (exp - 15u + 127u) & 0xFFu;
                    f32Bits = (sign << 31) | (fExp << 23) | (mant << 13);
                }
            }
            else if (exp == 0x1F)
            {
                f32Bits = (sign << 31) | 0x7F800000u | (mant << 13);
            }
            else
            {
                uint fExp = (exp - 15u + 127u) & 0xFFu;
                f32Bits = (sign << 31) | (fExp << 23) | (mant << 13);
            }
            return BitConverter.Int32BitsToSingle(unchecked((int)f32Bits));
        }
    }
}
