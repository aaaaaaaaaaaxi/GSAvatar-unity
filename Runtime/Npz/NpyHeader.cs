// NpyHeader.cs — parses a single NPY 1.0 / 2.0 / 3.0 header.
//
// Layout (after the 10-byte preamble):
//   { 'descr': '<f2', 'fortran_order': False, 'shape': (200000, 3), }
// followed by spaces and a single newline so the total header length
// (declared in bytes 8-9) is satisfied.
//
// We only support numeric descrs that show up in webavatar-rust NPZ files:
//   <u1 / <u2 / <u4 / <f2 / <f4 / >u1 / >u2 / >u4 / >f2 / >f4

using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace WebAvatar.Npz
{
    public enum NpyDType : byte
    {
        UInt8,
        UInt16,
        UInt32,
        Float16,
        Float32,
    }

    public sealed class NpyHeader
    {
        public NpyDType DType { get; }
        public int ElementSize { get; }
        public int[] Shape { get; }
        public int NumElements
        {
            get
            {
                if (Shape.Length == 0) return 1;
                long n = 1;
                for (int i = 0; i < Shape.Length; i++) n *= Shape[i];
                return (int)n;
            }
        }
        public int DataSize => NumElements * ElementSize;
        public bool IsLittleEndian { get; }

        public NpyHeader(NpyDType dtype, int elementSize, int[] shape, bool littleEndian)
        {
            DType = dtype;
            ElementSize = elementSize;
            Shape = shape;
            IsLittleEndian = littleEndian;
        }

        public const int PreambleSize = 10;

        // Magic + version + header_len read from a stream that has been
        // positioned at the start of an NPY entry. Returns the header length
        // and the data offset (== PreambleSize + headerLen).
        public static int ReadPreamble(ReadOnlySpan<byte> firstBytes, out int headerLen)
        {
            if (firstBytes.Length < PreambleSize)
                throw new InvalidOperationException("Buffer too small for NPY preamble");
            byte m0 = firstBytes[0], m1 = firstBytes[1];
            if (m0 != 0x93 || m1 != (byte)'N')
                throw new InvalidOperationException("Not an NPY file (magic mismatch)");
            // bytes 2-5 are 'U','M','P','Y'
            // bytes 6-7 are major,minor version; we accept any.
            headerLen = BinaryPrimitives.ReadUInt16LittleEndian(firstBytes.Slice(8, 2));
            return PreambleSize + headerLen;
        }

        // Parse the ASCII header dict that follows the preamble.
        public static NpyHeader Parse(ReadOnlySpan<byte> headerBytes)
        {
            // Skip any leading whitespace, then the '{', then split on commas
            // at the top level (we don't support nested dicts in numpy headers).
            string text = Encoding.ASCII.GetString(headerBytes).Trim();
            if (!text.StartsWith("{") || !text.EndsWith("}"))
                throw new InvalidOperationException("Malformed NPY header: " + text);

            var dict = ParseDict(text);

            string descr = dict["descr"].Trim('\'');
            bool littleEndian = true;
            char endian = descr[0];
            if (endian == '<' || endian == '|' ) littleEndian = true;
            else if (endian == '>') littleEndian = false;
            else throw new InvalidOperationException("Unsupported NPY descr endian: " + descr);

            string typeStr = descr.Substring(1);
            NpyDType dtype;
            int elementSize;
            switch (typeStr)
            {
                case "u1": dtype = NpyDType.UInt8; elementSize = 1; break;
                case "u2": dtype = NpyDType.UInt16; elementSize = 2; break;
                case "u4": dtype = NpyDType.UInt32; elementSize = 4; break;
                case "f2": dtype = NpyDType.Float16; elementSize = 2; break;
                case "f4": dtype = NpyDType.Float32; elementSize = 4; break;
                default:
                    throw new InvalidOperationException("Unsupported NPY dtype: " + descr);
            }

            int[] shape = ParseShape(dict["shape"]);
            return new NpyHeader(dtype, elementSize, shape, littleEndian);
        }

        static Dictionary<string, string> ParseDict(string text)
        {
            // text includes the surrounding braces — strip them if present.
            var result = new Dictionary<string, string>(StringComparer.Ordinal);
            int i = 0;
            // skip leading whitespace and the opening brace
            while (i < text.Length && char.IsWhiteSpace(text[i])) i++;
            if (i < text.Length && text[i] == '{') i++;
            while (i < text.Length)
            {
                // skip whitespace and commas
                while (i < text.Length && (char.IsWhiteSpace(text[i]) || text[i] == ',')) i++;
                if (i >= text.Length || text[i] == '}') break;

                // key (always single-quoted)
                if (text[i] != '\'') throw new InvalidOperationException("Expected ' at pos " + i);
                int keyStart = ++i;
                while (i < text.Length && text[i] != '\'') i++;
                if (i >= text.Length) throw new InvalidOperationException("Unterminated key");
                string key = text.Substring(keyStart, i - keyStart);
                i++; // closing quote

                // colon
                while (i < text.Length && char.IsWhiteSpace(text[i])) i++;
                if (i >= text.Length || text[i] != ':') throw new InvalidOperationException("Expected ':' at pos " + i);
                i++;
                while (i < text.Length && char.IsWhiteSpace(text[i])) i++;

                // value: either a quoted string or a tuple/list of numbers
                int valStart = i;
                if (text[i] == '\'')
                {
                    i++;
                    valStart = i;
                    while (i < text.Length && text[i] != '\'') i++;
                    if (i >= text.Length) throw new InvalidOperationException("Unterminated string value");
                    result[key] = text.Substring(valStart, i - valStart);
                    i++; // closing quote
                }
                else if (text[i] == '(' || text[i] == '[')
                {
                    char open = text[i];
                    char close = open == '(' ? ')' : ']';
                    i++;
                    int depth = 1;
                    while (i < text.Length && depth > 0)
                    {
                        if (text[i] == open) depth++;
                        else if (text[i] == close) depth--;
                        i++;
                    }
                    result[key] = text.Substring(valStart, i - valStart);
                }
                else
                {
                    // bare token (True/False/None or a number)
                    while (i < text.Length && text[i] != ',' && text[i] != '}' && !char.IsWhiteSpace(text[i])) i++;
                    result[key] = text.Substring(valStart, i - valStart);
                }
            }
            return result;
        }

        static int[] ParseShape(string tuple)
        {
            tuple = tuple.Trim();
            if (tuple == "()" || tuple == "[]") return Array.Empty<int>();
            char open = tuple[0];
            char close = open == '(' ? ')' : ']';
            string inner = tuple.Substring(1, tuple.Length - 2).Trim();
            if (inner.Length == 0) return Array.Empty<int>();
            // NumPy writes single-element shapes as "(200000,)" with a
            // trailing comma.  Splitting on ',' produces a trailing empty
            // entry; drop empties so int.Parse only sees real numbers.
            var parts = inner.Split(',');
            var shape = new int[parts.Length];
            int outIdx = 0;
            for (int i = 0; i < parts.Length; i++)
            {
                string token = parts[i].Trim();
                if (token.Length == 0) continue;
                shape[outIdx++] = int.Parse(token, CultureInfo.InvariantCulture);
            }
            if (outIdx != shape.Length)
                Array.Resize(ref shape, outIdx);
            return shape;
        }
    }
}
