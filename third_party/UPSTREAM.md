# Vendored third-party code

## third_party/vlmevalkit-cosmos

| field | value |
|---|---|
| upstream repo | https://github.com/NVIDIA/cosmos-framework |
| upstream path | `.agents/skills/evaluation/cosmos3/reasoner/vlmevalkit/` |
| upstream commit | `a3fb34690e157972a9d04244c11b16022c9d6479` ("Fix wsm ftheta derivative (#426)") |
| commit date | 2026-11-20 21:13:45 -0800 (2026-11-21 05:13 UTC) |
| split commit | `2a776869b204650c07588f8263c29cf2d1e9dd6f` (`git subtree split`; 8 upstream commits touch the path) |
| method | `git subtree split -P <path> -b vlmevalkit-cosmos` in a scratch clone (took ~1 s), then `git subtree add --prefix third_party/vlmevalkit-cosmos <scratch> vlmevalkit-cosmos --squash` (squash commit `28adaf7`, merge `718d5f1`) |
| vendored on | 2026-11-25 |
| license | Apache-2.0 (see `vlmevalkit-cosmos/LICENSE`) |

### Updating

```bash
git clone https://github.com/NVIDIA/cosmos-framework.git /tmp/cf && git -C /tmp/cf checkout <new-sha>
git -C /tmp/cf subtree split -P .agents/skills/evaluation/cosmos3/reasoner/vlmevalkit -b vlmevalkit-cosmos
git subtree pull --prefix third_party/vlmevalkit-cosmos /tmp/cf vlmevalkit-cosmos --squash
```
Then update this table.

### Local modifications (each in its own commit; see `git log -- third_party/vlmevalkit-cosmos`)

1. `vlmeval/api/cosmos_reason.py`, `vlmeval/api/__init__.py`: add `CosmosReason3Think`, a thinking
   wrapper that follows the official Cosmos 3 prompt guide (system prompt, `<think>` instruction,
   answer = text after the last `</think>`). Upstream classes are untouched. See PROTOCOL.md §4.
2. `vlmeval/api/cosmos_reason.py`: `CosmosReason3Think` also parses length-truncated replies (text after `</think>`,
   else empty prediction + `TRUNCATED` warning) instead of passing the raw reasoning to the scorer.
   See PROTOCOL.md §3.
