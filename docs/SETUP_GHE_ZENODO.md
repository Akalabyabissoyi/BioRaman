# Setup: MIB GitHub Enterprise + Zenodo DOI

The local repo is already initialised and tagged `v0.7.0`. These are the
remaining steps that need your MIB account and a Zenodo login — they can't be
done from here.

## 1. Push to MIB's GitHub Enterprise

1. On MIB's GHE (e.g. `github.mib.<org>.edu`), create a new **empty** repo
   named `raman-map-explorer` (no README/license — you already have them).
2. From this folder, add the remote and push, including the tag:

   ```bash
   git remote add origin git@<ghe-host>:<your-org-or-user>/raman-map-explorer.git
   git branch -M main
   git push -u origin main
   git push origin v0.7.0
   ```

   Use SSH (above) or HTTPS with a personal access token, depending on MIB's policy.

## 2. Connect Zenodo for a citable DOI

Zenodo's standard GitHub integration works with github.com. **Check whether
MIB's GHE is reachable by Zenodo** — many Enterprise instances are internal-only.

- **If the repo can live (or be mirrored) on github.com:**
  1. Log in to <https://zenodo.org> with the relevant GitHub account.
  2. Go to **Settings → GitHub**, find the repo, flip its toggle **On**.
  3. On GitHub, create a **Release** from tag `v0.7.0`. Zenodo automatically
     archives it and mints a DOI.
  4. Zenodo issues two DOIs: a **concept DOI** (always points to the latest
     version — cite this in papers) and a **version DOI** (this specific release).

- **If MIB requires the canonical repo to stay on GHE:**
  - Mirror to a github.com repo and connect Zenodo to the mirror, **or**
  - Use Zenodo's manual upload / API (`zenodo-cli`, or the deposit REST API)
    to archive each release tarball. The DOI is still valid for citation.

## 3. After the first DOI is minted

1. Add the concept DOI to `CITATION.cff` (uncomment the `doi:` line).
2. Add the Zenodo badge to the top of `README.md`:

   ```markdown
   [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
   ```
3. Commit, tag the next release, and push — Zenodo archives each new release automatically.

## Release workflow going forward (SemVer)

For every release that changes analytical output:

1. Update `CHANGELOG.md` (move items out of *Unreleased*, flag **[OUTPUT]** changes).
2. Bump `VERSION` and the `version:` field in `CITATION.cff`.
3. Commit, then `git tag -a vX.Y.Z -m "..."` and `git push --tags`.
4. Cut a GitHub Release from the tag → Zenodo mints a new version DOI.

Version bump rules:
- **MAJOR** — breaking changes to file formats or the public workflow.
- **MINOR** — new features, or any change that alters numerical results (**[OUTPUT]**).
- **PATCH** — bug fixes with no effect on output.
