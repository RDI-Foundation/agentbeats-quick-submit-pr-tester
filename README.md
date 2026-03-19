AgentBeats quick-submit end-to-end test leaderboard.

This repo is managed by the AgentBeats `e2e/quick-submit-pr` harness. It behaves like a green-agent leaderboard repo:

- quick-submit PRs run through the real upstream reusable workflow
- green and purple agent manifests are synced into this repo branch and point at deterministic debate agents running in public tutorial images
- `e2e-tunnel-probe.yml` verifies GitHub-hosted runners can reach the temporary backend and webhook tunnels
- merged quick-submit results land in `results/`
