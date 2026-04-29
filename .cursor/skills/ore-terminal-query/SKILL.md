---
name: ore-terminal-query
description: How to run ORE (Oxford Research Engine) from the terminal in this repo — correct `ore run` syntax, zsh quoting, and flags. Use when the user wants a research question formatted for the CLI or asks how to invoke `ore`.
---

# ORE terminal queries (this repo)

## If `zsh: command not found: ore`

The `ore` executable is registered in `pyproject.toml` as `[project.scripts]` — it only exists after you install this package into the **active Python environment**.

**Recommended (repo-local venv, avoids polluting global Python and avoids arm64/x86_64 wheel mixups on Apple Silicon):**

```bash
cd /path/to/oxford-research-assistant
./scripts/bootstrap.sh
source .venv/bin/activate
ore run 'Your question?' --rounds 4 --delay 12
```

Or without activating: `.venv/bin/ore run '...'`

**Global / user install** (only if you insist): `python3 -m pip install -e ".[search]"` — then put the env’s `bin` on `PATH` (e.g. `~/Library/Python/3.11/bin` is **not** on PATH by default on macOS).

### Apple Silicon: `mach-o ... incompatible architecture (have 'x86_64', need 'arm64')`

That means a **native arm64** Python is loading **Intel-only** wheels from an old site-packages tree (often `/Library/Frameworks/Python.framework/...`). **Use the repo `.venv`** from `./scripts/bootstrap.sh` so all wheels match one architecture, or create a fresh conda/mamba env on arm64.

### `SSL: CERTIFICATE_VERIFY_FAILED` / `unable to get local issuer certificate`

That is **your Python/OpenSSL trust store**, not ORE. Until PyPI is reachable over HTTPS, pip cannot download build tools (`setuptools`) or runtime deps (`litellm`, etc.).

Try in order:

1. **macOS, python.org installer:** run  
   `/Applications/Python\ 3.*/Install\ Certificates.command`  
   (pick the folder matching your Python version).

2. **Point Python at certifi** (after certifi exists in that env, or install via conda):
   ```bash
   export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
   python -m pip install -e .
   ```

3. **Conda:** `conda install ca-certificates pip` then retry; or create a fresh env from conda-forge.

4. **Corporate proxy / custom CA:** install the org root cert into the system or Python trust store (IT docs).

5. **Build deps already present:** avoid the isolated build env fetching PyPI:
   ```bash
   python -m pip install --no-build-isolation -e .
   ```
   (Requires a recent `setuptools`/`wheel` already installed in that interpreter.)

### Run without installing the `ore` script

If the package is installed but `ore` is not on `PATH`:

```bash
python -m ore run 'Your question?' --rounds 4 --delay 12
```

From a dev checkout **without** pip install (only if dependencies are available some other way):

```bash
cd /path/to/oxford-research-assistant
PYTHONPATH=src python -m ore run 'Your question?' --rounds 4 --delay 12
```

## Command shape

```bash
ore run '<QUESTION>' --rounds <N> --delay <SECONDS>
```

- **`ore run`** — entry point (see `src/ore/cli.py`).
- **`--rounds` / `-r`** — max research rounds (overrides config).
- **`--delay` / `-d`** — seconds between agent calls for rate-limit pacing (default **12**).
- **`--budget` / `-b`** — optional USD cap.
- **`--config` / `-c`** — YAML file instead of inline question.

Sessions save under `~/.ore/sessions/<id>/`.

## zsh: always single-quote the question

In **zsh**, `?`, `*`, and `[...]` inside a double-quoted or unquoted argument are **glob patterns**. A question like `...dimensions?` triggers `zsh: no matches found`.

**Fix:** wrap the entire question in **single quotes** so `?` is literal.

If the question contains an apostrophe (e.g. *don't*), you cannot use one continuous single-quoted string. Use one of:

1. **Break and concatenate:** `'...can'\''t...'` (end quote, escaped `'`, continue).
2. **Double quotes + disable glob for one command:** `noglob ore run "..."`  
3. **Here-doc:**

```bash
ore run "$(cat <<'EOF'
Your full question with ? and ' apostrophes as needed.
EOF
)" --rounds 4 --delay 12
```

## What to give the user

When they ask for “a question to ask ORE in terminal,” output:

1. The **full one-line** `ore run ...` command (not the bare question alone).
2. The question in **single-quoted** form unless it needs here-doc (apostrophes).
3. Suggested **`--rounds`** and **`--delay`** if they care about depth vs. pacing (e.g. `--rounds 4 --delay 12`).

## Optional: attach a local HTML (or text) file to a config run

In a YAML config next to `question:`, set:

```yaml
source_document: "/absolute/or/relative/path/to/file.html"
```

Paths are read by ORE when the session starts; the file is converted to plain text and appended to `question`. Relative paths are resolved from the YAML file’s directory. Hyperlinks and `file://` URLs are **not** fetched by the models — this is the supported way to “feed” a document.

## Reference

- Bootstrap: `scripts/bootstrap.sh`
- Usage and examples: `README.md` in repo root.
- CLI flags: `src/ore/cli.py` (`run` command).
