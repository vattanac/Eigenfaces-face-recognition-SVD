# Eigenfaces — Face Recognition with SVD / PCA

> Interactive browser visualization of the Eigenfaces algorithm using **real sklearn PCA** on the AT&T Olivetti Faces dataset — 400 faces, 40 identities, fully self-contained HTML output.

---

## Overview

This project implements the classic **Eigenfaces** face recognition technique (Turk & Pentland, 1991). A Python script runs `sklearn.decomposition.PCA` on the real AT&T Olivetti dataset, then bakes all results into a single self-contained HTML file that runs entirely in the browser — no server, no dependencies.

```
Python + sklearn  →  JSON / base64  →  eigenfaces_olivetti.html  →  Browser
```

---

## Demo

Open `eigenfaces_sklearn.html` directly — it contains the full interface pre-loaded with simulated data. No Python or setup required.

| Panel | What you see |
|---|---|
| **1 · Dataset** | 40 identity cards + full 400-face gallery |
| **2 · SVD** | Mean face · singular value chart · explained variance |
| **3 · Eigenfaces** | Top-k eigenfaces · cumulative variance curve |
| **4 · Reconstruct** | Original vs. reconstructed face pairs at current k |
| **5 · Recognize** | Noisy query → nearest-neighbor · top-8 L2 distances |

---

## How It Works

```
fetch_olivetti_faces()          X (400, 4096)
        ↓
PCA(n_components=50).fit(X)     components_, mean_, EVR, projections
        ↓
Downsample 64×64 → 32×32        Pillow / numpy fallback
        ↓
base64 encode all data          faces + eigenfaces + projections
        ↓
inject into template.html       replace /* __SKLEARN_DATA__ */ marker
        ↓
eigenfaces_olivetti.html        ~2.1 MB · fully self-contained
```

**Math:**

| Operation | Formula |
|---|---|
| Center data | `X_c = X − μ` |
| SVD | `X_c = U Σ Vᵀ` |
| Eigenfaces | top-k rows of `Vᵀ` → `pca.components_` |
| Project | `w = (x − μ) · Vᵏ` → `pca.transform(x)` |
| Reconstruct | `x̂ = w · Vᵏ + μ` → `pca.inverse_transform(w)` |
| Recognize | `argmin ‖W − w_query‖₂` |

---

## Quickstart

### Requirements

- Python 3.7+
- Git Bash (Windows) or any terminal

### Install

```bash
pip install scikit-learn numpy pillow
```

### Run

```bash
# Navigate to the project folder
cd /c/Users/YourName/Downloads/eigenfaces   # Git Bash (Windows)

# Generate the visualization
python generate_eigenfaces.py
```

### Open

```bash
start eigenfaces_olivetti.html   # Windows
open eigenfaces_olivetti.html    # macOS
```

Or just double-click the file in Explorer / Finder.

---

## Files

| File | Description |
|---|---|
| `generate_eigenfaces.py` | **Run this.** Downloads Olivetti data, runs sklearn PCA, builds HTML |
| `eigenfaces_template.html` | HTML shell with `/* __SKLEARN_DATA__ */` placeholder — keep alongside the script |
| `eigenfaces_sklearn.html` | Preview with simulated data — open immediately, no setup needed |
| `eigenfaces_olivetti.html` | **Generated output.** Real 400 Olivetti faces + sklearn PCA results embedded |
| `documents/index.html` | Full technical documentation and user guide |
| `readme-window/index.html` | Detailed Windows setup guide |

---

## Controls

| Control | Description |
|---|---|
| **k slider** (1–50) | Number of PCA components — affects reconstruction quality and recognition |
| **Noise slider** (0–60%) | Amount of noise added to query face before recognition |
| **Apply k & Refresh** | Recompute all panels with current k |
| **Recognize Query** | Run nearest-neighbor in eigenspace |
| **New Query Face** | Pick a random face as the query |
| Click any person card | Set that identity as the query subject |

---

## Dataset

**AT&T Olivetti Faces** — collected at AT&T Laboratories Cambridge, 1992–1994.

| Property | Value |
|---|---|
| Total images | 400 |
| Identities | 40 subjects |
| Images per person | 10 |
| Image size | 64 × 64 px · grayscale |
| Feature dimension | 4,096 |
| sklearn access | `fetch_olivetti_faces()` |

---

## Troubleshooting

**`python` is not recognized**
→ Try `python3`, or reinstall Python and check "Add Python to PATH"

**`pip` is not recognized**
→ Use `python -m pip install scikit-learn numpy pillow`

**Download error / figshare blocked**
→ The dataset is fetched from figshare.com (~5 MB, one time only). Try a different network or VPN. Cached after first run in `~/scikit_learn_data/`.

**`eigenfaces_template.html` not found**
→ Run `ls` — both `generate_eigenfaces.py` and `eigenfaces_template.html` must be in the same folder.

**Blank page in browser**
→ Use Chrome or Firefox. If the file is too large, serve locally:
```bash
python -m http.server 8080
# then open http://localhost:8080/eigenfaces_olivetti.html
```

---

## Tech Stack

- **Python** — `scikit-learn`, `numpy`, `Pillow`
- **Browser** — HTML5 Canvas, vanilla JavaScript (no frameworks)
- **Output** — fully self-contained single HTML file (no server, no CDN)

---

## References

- Turk, M. & Pentland, A. (1991). *Eigenfaces for Recognition*. Journal of Cognitive Neuroscience.
- Samaria, F. & Harter, A. (1994). *Parameterisation of a stochastic model for human face identification*. AT&T Laboratories Cambridge.
- [sklearn.decomposition.PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [sklearn Olivetti Faces dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html)

---

*MSDE Cohort 4 · Math Module · Eigenfaces / SVD Project*
