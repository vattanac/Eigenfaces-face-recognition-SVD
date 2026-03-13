#!/usr/bin/env python3
"""
generate_eigenfaces.py — Real sklearn PCA on Olivetti Faces → HTML

Usage:
    pip install scikit-learn numpy pillow
    python3 generate_eigenfaces.py
    open eigenfaces_olivetti.html
"""
import sys, os, json, base64
import numpy as np

print("="*62)
print("  Eigenfaces: Real sklearn PCA + Olivetti Faces → HTML")
print("="*62)

# 1. Fetch real dataset
print("\n[1/5] Fetching Olivetti faces via sklearn...")
try:
    from sklearn.datasets import fetch_olivetti_faces
    from sklearn.decomposition import PCA
except ImportError:
    print("ERROR: pip install scikit-learn numpy pillow"); sys.exit(1)

dataset = fetch_olivetti_faces(shuffle=False)
X, y, images = dataset.data, dataset.target, dataset.images
print(f"     X.shape={X.shape}  y.shape={y.shape}  images.shape={images.shape}")

# 2. Real sklearn PCA
print("\n[2/5] Running PCA(n_components=50).fit(X)...")
K = 50
pca = PCA(n_components=K, whiten=False)
pca.fit(X)
eigenfaces_full = pca.components_
mean_face_full  = pca.mean_
evr             = pca.explained_variance_ratio_
cum_evr         = evr.cumsum()
singular_v      = np.sqrt(pca.explained_variance_ * (len(X)-1))
projections     = pca.transform(X)
print(f"     components_.shape={eigenfaces_full.shape}  EVR[0]={evr[0]:.4f}  cumEVR@k10={cum_evr[9]:.3f}")

# 3. Downsample 64x64 -> 32x32
print("\n[3/5] Downsampling 64x64 -> 32x32...")
try:
    from PIL import Image as PILImage
    def resize32(a): return np.array(PILImage.fromarray((a.clip(0,1)*255).astype('uint8'),'L').resize((32,32),PILImage.LANCZOS),dtype='float32')/255
except ImportError:
    def resize32(a): return a.reshape(32,2,32,2).mean(axis=(1,3)).astype('float32')

faces_32 = np.stack([resize32(images[i]) for i in range(400)])
mean_32  = resize32(mean_face_full.reshape(64,64))
ef_32    = np.stack([resize32(((eigenfaces_full[i].reshape(64,64)-eigenfaces_full[i].min())/(eigenfaces_full[i].max()-eigenfaces_full[i].min()+1e-8)).astype('float32')) for i in range(K)])

# 4. Encode
print("\n[4/5] Encoding to base64...")
def b64(a): return base64.b64encode((a.clip(0,1)*255).astype('uint8').tobytes()).decode('ascii')

payload = {
    "faces_b64":   b64(faces_32.reshape(400,-1)),
    "mean_b64":    b64(mean_32.flatten()),
    "ef_b64":      b64(ef_32.reshape(K,-1)),
    "projections": projections.tolist(),
    "evr":         evr.tolist(),
    "cum_evr":     cum_evr.tolist(),
    "singular_v":  singular_v.tolist(),
    "labels":      y.tolist(),
    "k":K,"n":400,"img_w":32,"img_h":32,"n_ids":40,
}
json_str = json.dumps(payload, separators=(',',':'))
print(f"     Payload: {len(json_str)//1024} KB")

# 5. Inject into template
print("\n[5/5] Building eigenfaces_olivetti.html...")
tmpl = os.path.join(os.path.dirname(__file__), 'eigenfaces_template.html')
out  = os.path.join(os.path.dirname(__file__), 'eigenfaces_olivetti.html')

if not os.path.exists(tmpl):
    print("ERROR: eigenfaces_template.html not found in same folder."); sys.exit(1)

html = open(tmpl).read().replace('/* __SKLEARN_DATA__ */', f'const DATA = {json_str};')
open(out,'w').write(html)

print(f"\n{'='*62}")
print(f"  ✓  eigenfaces_olivetti.html  ({os.path.getsize(out)//1024} KB)")
print(f"{'='*62}")
print(f"\n  Baked in:")
print(f"    pca.mean_                    → mean face (32x32)")
print(f"    pca.components_[:50]         → 50 eigenfaces")
print(f"    pca.transform(X)             → 400x50 projections")
print(f"    pca.explained_variance_ratio_ → variance spectrum")
print(f"\n  Open:  open {out}")
