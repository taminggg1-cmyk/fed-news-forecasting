# Pre-registrations

Each file was committed and tagged before the results it governs. The tag is the authoritative record:
`git show <tag>:<file>` shows the file as registered (at the repository root in the tagged commit).

| File | Hypotheses | Tag | Commit, time (UTC+7) |
| --- | --- | --- | --- |
| `PREREG.md` | H1 (statistics vs statistics + news; linear models). The secondary H2 family (H2a-H2g) is defined in the design spec `docs/superpowers/specs/2026-10-08-fed-news-design.md`, committed under the same tag | `prereg` | 5d0c31c, 2026-10-09 13:00 |
| `PREREG_H2A.md` | H2a (paper-style LSTM / GRU, TFT, hybrid) | `prereg-h2a` | e931dfc, 2026-10-09 17:44 |
| `PREREG_TEXT.md` | H5 (event study of the 2-year yield) | `prereg-text` | 8d2c5c1, 2026-10-10 14:12 |
| `PREREG_STUDY2.md` | H6 (event study with regional Fed speeches) | `prereg-study2` | b967d17, 2026-10-10 16:48 |

Studies 3, 4 and 5 and the Study 4 version 2 networks were not registered with a tag. Their designs were committed
before their runs (`docs/superpowers/specs/`); the version 2 network protocol was written after the version 1
results had been seen (commit 6048e0d). Every later change is dated in [`../../DEVIATIONS.md`](../../DEVIATIONS.md).
