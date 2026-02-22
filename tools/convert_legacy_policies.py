#!/usr/bin/env python3
"""One-time converter for legacy policy assets under data/policies/**.

By default this runs in dry-run mode and only writes a manifest.
Use --apply to rewrite arg keys and create .pt copies next to legacy .h5 models.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def convert_args(repo_root: Path, apply: bool) -> list[dict]:
    updates = []
    for arg_file in sorted((repo_root / "args").glob("*.txt")):
        text = arg_file.read_text()
        new_text = text.replace("-policy_net=", "-policy_arch_config=").replace(
            "-policy_solver=", "-policy_checkpoint="
        )
        if new_text != text:
            if apply:
                arg_file.write_text(new_text)
            updates.append(
                {
                    "file": str(arg_file.relative_to(repo_root)),
                    "type": "arg_keys",
                    "applied": apply,
                }
            )
    return updates


def convert_models(repo_root: Path, apply: bool) -> list[dict]:
    updates = []
    for h5_file in sorted((repo_root / "data" / "policies").glob("**/*.h5")):
        pt_file = h5_file.with_suffix(".pt")
        if not pt_file.exists():
            if apply:
                shutil.copyfile(h5_file, pt_file)
            updates.append(
                {
                    "source": str(h5_file.relative_to(repo_root)),
                    "target": str(pt_file.relative_to(repo_root)),
                    "type": "checkpoint_copy",
                    "applied": apply,
                }
            )
    return updates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Apply conversion changes in-place")
    parser.add_argument(
        "--manifest",
        default="output/legacy_policy_conversion_manifest.json",
        help="Where to write the conversion manifest",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    arg_updates = convert_args(repo_root, args.apply)
    model_updates = convert_models(repo_root, args.apply)

    manifest = {
        "description": "One-time legacy policy conversion manifest",
        "dry_run": not args.apply,
        "arg_updates": arg_updates,
        "model_updates": model_updates,
    }

    manifest_path = repo_root / args.manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] Wrote manifest: {manifest_path}")
    print(f"Arg files to update: {len(arg_updates)}")
    print(f"Model checkpoints to copy: {len(model_updates)}")


if __name__ == "__main__":
    main()
