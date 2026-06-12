// WebAvatarAssetImporter.cs — turns a .npz into a WebAvatarAsset.
//
// This is a thin ScriptedImporter; all real work is in
// AvatarDataBuilder so it can be unit-tested outside the asset DB.

using System.IO;
using UnityEditor.AssetImporters;
using UnityEngine;

namespace WebAvatar.Editor
{
    [ScriptedImporter(1, "npz")]
    public class WebAvatarAssetImporter : ScriptedImporter
    {
        public override void OnImportAsset(AssetImportContext ctx)
        {
            var asset = AvatarDataBuilder.Build(ctx.assetPath);
            asset.name = Path.GetFileNameWithoutExtension(ctx.assetPath);

            ctx.AddObjectToAsset("WebAvatarAsset", asset);
            ctx.SetMainObject(asset);
        }
    }
}
