"""Audit tracked/public source without printing matching secret values."""

import argparse
import ipaddress
import json
import re
import subprocess
from pathlib import Path

RULES = {
    "credential": re.compile(
        r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)"
    ),
    "machine_path": re.compile(r"/(?:home|root|mnt|cpfs|Knowin)/(?:[A-Za-z0-9_.-]+/)+"),
    "private_endpoint": re.compile(
        r"https?://[^\s\"'<>]+(?:\.internal|\.local|\.lan)(?:[:/\s]|$)", re.I
    ),
    "embedded_auth": re.compile(r"https?://[^\s/@]+:[^\s/@]+@"),
}
FORBIDDEN = {".env", ".DS_Store", "Thumbs.db", "id_rsa", "id_ed25519"}
SUFFIXES = {".pyc", ".log", ".pt", ".pth", ".safetensors", ".ckpt", ".key", ".pem"}


def scan(root, tracked=True):
    if tracked:
        result = subprocess.run(
            ["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True
        )
        paths = [root / p for p in result.stdout.decode().split("\0") if p]
    else:
        paths = [
            p
            for p in root.rglob("*")
            if p.is_file()
            and not any(
                x in {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}
                or x.endswith(".egg-info")
                for x in p.relative_to(root).parts
            )
        ]
    findings = []
    for path in paths:
        rel = str(path.relative_to(root))
        if path.is_symlink():
            findings.append({"file": rel, "rule": "symlink"})
            continue
        if (
            path.name in FORBIDDEN
            or path.suffix in SUFFIXES
            or any(
                x in {"logs", "wandb", ".cache", "checkpoints", "predictions"}
                for x in path.relative_to(root).parts
            )
        ):
            findings.append({"file": rel, "rule": "runtime_or_sensitive_artifact"})
        data = path.read_bytes()
        if len(data) > 10 * 1024 * 1024:
            findings.append({"file": rel, "rule": "large_file"})
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in RULES.items():
            if pattern.search(text):
                findings.append({"file": rel, "rule": label})
        for value in re.findall(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])", text):
            try:
                ip = ipaddress.ip_address(value)
            except ValueError:
                continue
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                findings.append({"file": rel, "rule": "nonpublic_ip"})
    return {"file_count": len(paths), "findings": findings}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=".")
    p.add_argument("--all-files", action="store_true")
    args = p.parse_args()
    result = scan(Path(args.root).resolve(), not args.all_files)
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(result["findings"]))


if __name__ == "__main__":
    main()
