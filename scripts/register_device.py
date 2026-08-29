#!/usr/bin/env python3
"""One-time reMarkable device registration.

Run locally, ONCE. Writes the device token straight into SSM SecureString so it
never touches a file that could be committed.

    1. Open https://my.remarkable.com/device/desktop/connect
    2. Copy the eight-character code
    3. python scripts/register_device.py --code ABCDEFGH --profile your-aws-profile

The device token is long-lived. The short-lived user token used for each API
call is derived from it at runtime and never stored.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "functions"))

from rmsync.auth import register_device  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", required=True, help="8-character code from my.remarkable.com")
    parser.add_argument("--parameter", default="/rmsync/remarkable-token")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the token instead of writing it to SSM (avoid: it lands in shell history)",
    )
    args = parser.parse_args()

    print("Registering device with reMarkable Cloud...")
    token = register_device(args.code)

    if args.print_only:
        print(token)
        return 0

    cmd = [
        "aws", "ssm", "put-parameter",
        "--name", args.parameter,
        "--type", "SecureString",
        "--value", token,
        "--overwrite",
        "--region", args.region,
    ]
    if args.profile:
        cmd += ["--profile", args.profile]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Failed to write {args.parameter}:\n{result.stderr}", file=sys.stderr)
        return 1
    print(f"Device token stored at {args.parameter} (SecureString).")
    print("This token is long-lived - you should not need to run this again.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
