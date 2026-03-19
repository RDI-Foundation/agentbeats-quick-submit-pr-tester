AgentBeats quick-submit end-to-end test leaderboard.

This repo is managed by the AgentBeats `e2e/quick-submit-pr` harness. It behaves like a green-agent leaderboard repo:

- quick-submit PRs run through the real upstream reusable workflow
- mock green and purple manifests live under `manifests/`
- `e2e-tunnel-probe.yml` verifies GitHub-hosted runners can reach the temporary backend and webhook tunnels
- merged quick-submit results land in `results/`
