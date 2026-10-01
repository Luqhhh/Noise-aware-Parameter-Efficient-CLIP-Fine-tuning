"""Export or verify the explicit, checksummed three-machine handoff whitelist."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = ROOT / "configs/team_exploration_20261001/asset_sources.json"


def digest_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def checked_name(name):
    path = PurePosixPath(name)
    if (not name or "\\" in name or path.is_absolute() or ".." in path.parts
            or name != path.as_posix() or not name.startswith("assets/")):
        raise ValueError(f"Invalid handoff member: {name}")
    return name


def load_sources(path):
    config = json.loads(Path(path).read_text())
    if config["schema"] != 1 or config["data_version"] != "20260921":
        raise ValueError("Unexpected handoff stage or schema")
    seen = set()
    for item in config["files"]:
        name = checked_name(item["archive_path"])
        if name in seen:
            raise ValueError("Duplicate handoff member")
        seen.add(name)
        source = Path(item["source_path"])
        if (not source.is_file() or source.stat().st_size != item["bytes"]
                or digest_file(source) != item["sha256"]):
            raise ValueError(f"Whitelisted source changed or unavailable: {name}")
    return config


def export(sources, output):
    config = load_sources(sources)
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Never replace a colleague's archive, including on a failed rerun.
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        archive.writestr("manifest.json", json.dumps(config, ensure_ascii=False, indent=2) + "\n")
        for item in config["files"]:
            archive.write(item["source_path"], arcname=item["archive_path"])
    return verify(output, sources)


def verify(bundle, sources):
    # The git-tracked source whitelist anchors hashes; a self-consistent forged
    # ZIP manifest alone is insufficient. Local source paths need not exist.
    expected = json.loads(Path(sources).read_text())
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest != expected:
            raise ValueError("ZIP manifest differs from the reviewed source whitelist")
        names = [item["archive_path"] for item in manifest["files"]]
        for name in names:
            checked_name(name)
        members = archive.namelist()
        if len(set(members)) != len(members) or set(members) != {"manifest.json", *names}:
            raise ValueError("Missing, extra or duplicate ZIP member")
        for item in manifest["files"]:
            result = hashlib.sha256()
            size = 0
            with archive.open(item["archive_path"]) as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    result.update(chunk)
                    size += len(chunk)
            if result.hexdigest() != item["sha256"] or size != item["bytes"]:
                raise ValueError(f"Corrupt handoff asset: {item['archive_path']}")
    return dict(status="verified", files=len(names), data_version=manifest["data_version"],
                bundle=str(Path(bundle).resolve()), bundle_sha256=digest_file(bundle),
                bundle_bytes=Path(bundle).stat().st_size,
                training_started=False, gpu_started=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "verify"))
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()
    if args.action == "export":
        if not args.output:
            parser.error("export requires a fresh --output")
        report = export(args.sources, args.output)
    else:
        if not args.bundle:
            parser.error("verify requires --bundle")
        report = verify(args.bundle, args.sources)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
