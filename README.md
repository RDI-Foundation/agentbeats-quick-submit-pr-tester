AgentBeats quick-submit end-to-end test leaderboard.

This repo is managed by the AgentBeats `e2e/quick-submit-pr` harness. It behaves like a green-agent leaderboard repo:

- quick-submit PRs run through the real upstream reusable workflow
- mock green and purple agent images are published to GHCR
- merged quick-submit results land in `results/`
