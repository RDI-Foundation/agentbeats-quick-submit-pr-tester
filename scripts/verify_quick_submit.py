#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for older Python
    import tomli as tomllib  # type: ignore


SECRET_PATTERN = re.compile(r"\$\{([^}]+)\}")


def extract_secret_names(env_dict: dict) -> set[str]:
    names: set[str] = set()
    for value in env_dict.values():
        for match in SECRET_PATTERN.findall(str(value)):
            names.add(match)
    return names


def parse_scenario(path: Path) -> tuple[set[str], list[dict]]:
    data = tomllib.loads(path.read_text())

    green = data.get("green_agent", {})
    green_id = green.get("agentbeats_id")
    if not green_id:
        raise SystemExit("scenario.toml missing green_agent.agentbeats_id")
    green_secrets = extract_secret_names(green.get("env") or {})

    participants = []
    for item in data.get("participants") or []:
        agent_id = item.get("agentbeats_id")
        name = item.get("name")
        if not agent_id or not name:
            raise SystemExit("scenario.toml participant missing agentbeats_id or name")
        secrets = extract_secret_names(item.get("env") or {})
        participants.append(
            {
                "agent_id": str(agent_id),
                "name": str(name),
                "secrets": secrets,
            }
        )

    return green_secrets, participants


def validate_green(expected: set[str], green_blob: dict) -> dict:
    if not isinstance(green_blob, dict):
        raise SystemExit("Quick submit secrets missing green object")

    keys = set(green_blob.keys())
    missing = expected - keys
    extra = keys - expected
    if missing:
        raise SystemExit(f"Missing green secrets: {', '.join(sorted(missing))}")
    if extra:
        raise SystemExit(f"Unexpected green secrets: {', '.join(sorted(extra))}")
    for key in expected:
        value = green_blob.get(key)
        if value is None or value == "":
            raise SystemExit(f"Green secret {key} is empty")
    return green_blob


def participant_key(agent_id: str, name: str) -> str:
    return f"{agent_id}:{name}"


def validate_participants(expected: list[dict], participants_blob: list) -> dict[str, dict]:
    if not isinstance(participants_blob, list):
        raise SystemExit("Quick submit secrets missing participants list")

    secrets_participants: dict[str, dict] = {}
    for item in participants_blob:
        if not isinstance(item, dict):
            raise SystemExit("Invalid participant entry in quick submit secrets payload")
        agent_id = str(item.get("agent_id") or "")
        name = str(item.get("name") or "")
        if not agent_id or not name:
            raise SystemExit("Participant entry missing agent_id or name in secrets payload")
        key = participant_key(agent_id, name)
        if key in secrets_participants:
            raise SystemExit(f"Duplicate participant entry in secrets payload: {key}")
        secrets_participants[key] = item

    for participant in expected:
        key = participant_key(participant["agent_id"], participant["name"])
        secrets_entry = secrets_participants.get(key)
        if secrets_entry is None:
            raise SystemExit(f"Missing quick submit secrets for participant {key}")
        secrets_map = secrets_entry.get("secrets") or {}
        if not isinstance(secrets_map, dict):
            raise SystemExit(f"Invalid secrets map for participant {key}")
        expected_keys = set(participant["secrets"])
        actual_keys = set(secrets_map.keys())
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        if missing:
            raise SystemExit(
                f"Missing secrets for participant {key}: {', '.join(sorted(missing))}"
            )
        if extra:
            raise SystemExit(
                f"Unexpected secrets for participant {key}: {', '.join(sorted(extra))}"
            )
        for key_name in expected_keys:
            value = secrets_map.get(key_name)
            if value is None or value == "":
                raise SystemExit(f"Participant {key} secret {key_name} is empty")

    extras = set(secrets_participants.keys()) - {
        participant_key(p["agent_id"], p["name"]) for p in expected
    }
    if extras:
        raise SystemExit(
            "Quick submit secrets include participants not in scenario: "
            + ", ".join(sorted(extras))
        )

    return {
        participant_key(p["agent_id"], p["name"]): secrets_participants[
            participant_key(p["agent_id"], p["name"])
        ]
        for p in expected
    }


def run_checker(
    image: str,
    role: str,
    expected_keys: set[str],
    all_keys: set[str],
    secrets_map: dict,
) -> None:
    env = [
        "ROLE=" + role,
        "EXPECTED_SECRETS=" + ",".join(sorted(expected_keys)),
        "ALL_SECRETS=" + ",".join(sorted(all_keys)),
    ]
    for key in sorted(expected_keys):
        env.append(f"{key}={secrets_map.get(key, '')}")

    cmd = ["docker", "run", "--rm"]
    for entry in env:
        cmd.extend(["-e", entry])
    cmd.append(image)

    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate quick submit secrets and run checks")
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--secrets-file", type=Path, required=True)
    parser.add_argument("--image", default="secret-checker:local")
    args = parser.parse_args()

    green_required, participants_required = parse_scenario(args.scenario)
    secrets_blob = json.loads(args.secrets_file.read_text())

    green_blob = validate_green(green_required, secrets_blob.get("green") or {})
    participants_blob = validate_participants(
        participants_required, secrets_blob.get("participants") or []
    )

    all_secrets = set(green_required)
    for participant in participants_required:
        all_secrets.update(participant["secrets"])

    run_checker(
        args.image,
        "green-agent",
        green_required,
        all_secrets,
        green_blob,
    )

    for participant in participants_required:
        key = participant_key(participant["agent_id"], participant["name"])
        secrets_map = (participants_blob.get(key) or {}).get("secrets") or {}
        run_checker(
            args.image,
            f"participant-{participant['name']}",
            participant["secrets"],
            all_secrets,
            secrets_map,
        )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
