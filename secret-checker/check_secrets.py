import os
import sys


def split_env_list(name: str) -> list[str]:
    raw = os.environ.get(name, "")
    if not raw:
        return []
    return [item for item in raw.split(",") if item]


expected = set(split_env_list("EXPECTED_SECRETS"))
all_secrets = set(split_env_list("ALL_SECRETS"))
role = os.environ.get("ROLE", "unknown")

missing = [key for key in sorted(expected) if not os.environ.get(key)]
unexpected = [key for key in sorted(all_secrets - expected) if os.environ.get(key)]

if missing or unexpected:
    if missing:
        print(f"[{role}] missing secrets: {', '.join(missing)}", file=sys.stderr)
    if unexpected:
        print(f"[{role}] unexpected secrets: {', '.join(unexpected)}", file=sys.stderr)
    sys.exit(1)

print(f"[{role}] secrets validated: {', '.join(sorted(expected))}")
