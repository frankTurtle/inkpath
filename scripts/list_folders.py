#!/usr/bin/env python3
"""List your reMarkable folders so you can pick WatchFolder / WatchFolderId.

Read-only: it lists the library and prints nothing back to reMarkable.

    .venv/bin/python scripts/list_folders.py --profile your-aws-profile

Folder names are not unique in reMarkable - two folders can share a name under
different parents. Where that happens this prints DUPLICATE, and you should set
WatchFolderId to the specific id rather than matching by name.
"""

from __future__ import annotations

import argparse
import collections
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "functions"))

try:
    from rmsync.auth import get_secret, get_user_token
    from rmsync.remarkable import TYPE_COLLECTION, RemarkableClient
    from rmsync.scope import valid_parent_ids
except ModuleNotFoundError as exc:  # pragma: no cover - operator-facing guidance
    root = pathlib.Path(__file__).resolve().parents[1]
    venv = root / ".venv" / "bin" / "python"
    sys.exit(
        f"Missing dependency: {exc.name}\n\n"
        "Run this with the project virtualenv:\n\n"
        f"    {venv} {' '.join(sys.argv)}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--parameter", default="/rmsync/remarkable-token")
    parser.add_argument(
        "--show-documents",
        action="store_true",
        help="Also list document names inside each folder",
    )
    args = parser.parse_args()

    if args.profile:
        os.environ["AWS_PROFILE"] = args.profile
    os.environ.setdefault("AWS_DEFAULT_REGION", args.region)

    print("Reading device token from SSM...")
    device_token = get_secret(args.parameter)
    print("Exchanging for a user token...")
    user_token = get_user_token(device_token)
    print("Listing library (this walks the whole root index)...\n")

    client = RemarkableClient(user_token)
    items, _cache = client.list_items()

    folders = [i for i in items if i.type == TYPE_COLLECTION and not i.in_trash]
    docs = [i for i in items if i.is_document and not i.in_trash and not i.deleted]

    by_parent: dict[str, list] = collections.defaultdict(list)
    for doc in docs:
        by_parent[doc.parent].append(doc)

    name_counts = collections.Counter(f.visible_name.strip().lower() for f in folders)
    names = {f.id: f.visible_name for f in folders}

    print(f"{len(folders)} folder(s), {len(docs)} document(s)\n")
    print(f"{'FOLDER':<34} {'DIRECT':>6} {'NESTED':>7}  ID")
    print("-" * 92)

    for folder in sorted(folders, key=lambda f: f.visible_name.lower()):
        scoped = valid_parent_ids(items, folder.id)
        nested = sum(len(by_parent.get(pid, [])) for pid in scoped)
        direct = len(by_parent.get(folder.id, []))
        parent = names.get(folder.parent, "" if not folder.parent else "?")
        label = folder.visible_name + (f"  (in {parent})" if parent else "")
        dupe = "  << DUPLICATE NAME" if name_counts[folder.visible_name.strip().lower()] > 1 else ""
        print(f"{label[:33]:<34} {direct:>6} {nested:>7}  {folder.id}{dupe}")
        if args.show_documents:
            for doc in sorted(by_parent.get(folder.id, []), key=lambda d: d.visible_name.lower()):
                print(f"    - {doc.visible_name}")

    root_docs = by_parent.get("", [])
    if root_docs:
        print(f"\n{len(root_docs)} document(s) sit at the library root (no folder).")
        if args.show_documents:
            for doc in sorted(root_docs, key=lambda d: d.visible_name.lower()):
                print(f"    - {doc.visible_name}")

    print(
        "\nNEXT: pick a folder above. 'NESTED' is how many documents would be in\n"
        "scope, counting subfolders. Deploy with:\n"
        "    WatchFolder='<name>'          (or WatchFolderId=<id> if marked DUPLICATE)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
