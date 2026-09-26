# Notes ("memories")

This section is for **hard-fought findings**: the thing that cost an
afternoon, the chip quirk that is not in the manual, the training run that
diverged and why, the measurement that contradicted the paper. Anything a
future session (human or agent) would otherwise have to rediscover.

The notes themselves are **gitignored for now**. Only this index is
tracked. When served locally (`just docs`) they appear under this section
and in search, and the published site shows only this page. When the
project is ready to publish them, remove the `docs/notes/*` line from
`.gitignore` and add them to the nav in `zensical.toml`.

## Conventions

- One finding per file, named `YYYY-MM-DD-<slug>.md`.
- Start with the one-sentence takeaway in bold, then the evidence, then
  what to do about it.
- Cite the commit, the log file on the data drive, or the measurement that
  proves it. A note without evidence is a rumour.
- If a note turns out to be wrong, edit it to say so rather than deleting it.
  The correction is the valuable part.

## Where the data lives

Training data never enters the repository. It lives under the data root
(`$UNAMBLIFY_DATA`, by default `/Volumes/data/training_data/unamblify` on
an external drive). Logs of every fetch and capture run are kept under
`logs/` there.
