"""Compile all .tex files in autopath/docs using tectonic."""

import glob
import os
import subprocess
import sys


def main():
    docs_dir = os.path.join(os.path.dirname(__file__), "docs")
    tex_files = sorted(glob.glob(os.path.join(docs_dir, "**", "*.tex"), recursive=True))

    if not tex_files:
        print(f"No .tex files found in {docs_dir}")
        sys.exit(0)

    failed = []
    for tex_file in tex_files:
        rel = os.path.relpath(tex_file, docs_dir)
        print(f"\n{'='*60}")
        print(f"Compiling {rel}")
        print(f"{'='*60}")
        result = subprocess.run(
            ["tectonic", tex_file],
            cwd=os.path.dirname(tex_file),
        )
        if result.returncode != 0:
            failed.append(rel)

    print(f"\n{'='*60}")
    if failed:
        print(f"FAILED ({len(failed)}/{len(tex_files)}):")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print(f"All {len(tex_files)} documents compiled successfully.")


if __name__ == "__main__":
    main()
