// NpzArchive.cs — minimal NPZ reader.
//
// NPZ is just a ZIP archive of NPY files.  Each entry name is the array
// key in the originating Python savez(); for webavatar-rust the keys are
// things like "_xyz", "_sh0", "layers.0.weight", etc.
//
// We build everything on top of System.IO.Compression (built into the
// .NET Standard 2.0 / 2.1 surface that Unity ships), so we don't depend
// on Unity.IO.Compression or a third-party ZIP library.

using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;

namespace WebAvatar.Npz
{
    public sealed class NpzArchive : IDisposable
    {
        readonly ZipArchive _zip;
        readonly Dictionary<string, ZipArchiveEntry> _entries;
        bool _disposed;

        NpzArchive(ZipArchive zip)
        {
            _zip = zip;
            _entries = new Dictionary<string, ZipArchiveEntry>(zip.Entries.Count, StringComparer.Ordinal);
            foreach (var entry in zip.Entries)
            {
                // Strip trailing ".npy" — NumPy adds the extension on save.
                string name = entry.FullName;
                if (name.EndsWith(".npy", StringComparison.Ordinal))
                    name = name.Substring(0, name.Length - 4);
                _entries[name] = entry;
            }
        }

        public static NpzArchive Open(string path)
        {
            var fs = File.OpenRead(path);
            try
            {
                return new NpzArchive(new ZipArchive(fs, ZipArchiveMode.Read));
            }
            catch
            {
                fs.Dispose();
                throw;
            }
        }

        public static NpzArchive Open(Stream stream)
        {
            // The caller owns the stream; we don't take ownership here.
            return new NpzArchive(new ZipArchive(stream, ZipArchiveMode.Read));
        }

        public ICollection<string> Names => _entries.Keys;

        public bool Contains(string name) => _entries.ContainsKey(name);

        // Read a single NPY array.  Throws if the key is missing — use
        // TryGetArray to probe for optional entries.
        public NpyArray GetArray(string name)
        {
            if (!TryGetArray(name, out var arr))
                throw new KeyNotFoundException("NPZ entry not found: " + name);
            return arr;
        }

        public bool TryGetArray(string name, out NpyArray array)
        {
            array = null;
            if (!_entries.TryGetValue(name, out var entry)) return false;
            array = ReadEntry(name, entry);
            return true;
        }

        public byte[] GetRaw(string name)
        {
            if (!_entries.TryGetValue(name, out var entry))
                throw new KeyNotFoundException("NPZ entry not found: " + name);
            using var s = entry.Open();
            using var ms = new MemoryStream(checked((int)entry.Length));
            s.CopyTo(ms);
            return ms.ToArray();
        }

        static NpyArray ReadEntry(string name, ZipArchiveEntry entry)
        {
            using var s = entry.Open();

            // The NPY preamble is exactly 10 bytes.  Reading the whole stream
            // into memory is fine — every entry in webavatar-rust NPZ files
            // is at most a few tens of MB (xyz is 200k * 3 * 2 = 1.2 MB,
            // MLP layers are the next-biggest at ~600 KB each).
            using var ms = new MemoryStream(checked((int)entry.Length));
            s.CopyTo(ms);
            var bytes = ms.ToArray();

            int headerLen;
            int dataOffset = NpyHeader.ReadPreamble(bytes, out headerLen);
            if (dataOffset > bytes.Length)
                throw new InvalidDataException("NPY header truncated for entry '" + name + "'");

            var header = NpyHeader.Parse(new ReadOnlySpan<byte>(bytes, NpyHeader.PreambleSize, headerLen));
            var dataLen = bytes.Length - dataOffset;
            if (dataLen != header.DataSize)
                throw new InvalidDataException(
                    $"NPY data length {dataLen} != header.DataSize {header.DataSize} for entry '{name}'");

            // Copy the data slice so the NpyArray owns its buffer independently
            // of the (soon-to-be-collected) ZipArchiveEntry decode buffer.
            var payload = new byte[dataLen];
            Buffer.BlockCopy(bytes, dataOffset, payload, 0, dataLen);
            return new NpyArray(name, header, payload);
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _zip.Dispose();
        }
    }
}
