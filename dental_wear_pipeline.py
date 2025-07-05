
# dental_wear_pipeline.py

import os
import numpy as np
import pandas as pd
import trimesh
from scipy.spatial.distance import directed_hausdorff

# -------------------------
# Configuration
# -------------------------
MESH_DIR = "3D_scans_per_patient_obj_files"
PATIENT_IDS = [
    "01M6GFPV", "01KY7E6A", "01E5XG8Z", "01J9RWK6",  
]

# -------------------------
# Utilities
# -------------------------
def load_mesh(patient_id):
    path = os.path.join(MESH_DIR, patient_id, f"{patient_id}_upper.obj")
    return trimesh.load(path)

def simulate_wear_trimesh(mesh, z_shift=0.2, percent=0.1):
    mesh_worn = mesh.copy()
    vertices = mesh_worn.vertices.copy()
    z_thresh = np.percentile(vertices[:, 2], percent * 100)
    mask = vertices[:, 2] < z_thresh
    vertices[mask, 2] -= z_shift
    mesh_worn.vertices = vertices
    return mesh_worn

def sample_points(mesh, n=1000):
    return mesh.sample(n)

def estimate_curvature(points, k=10):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k).fit(points)
    _, idx = nn.kneighbors(points)
    curvature = []
    for i in range(len(points)):
        neighbors = points[idx[i][1:]]
        cov = np.cov(neighbors.T)
        eigvals = np.linalg.eigvalsh(cov)
        curvature.append(eigvals[0] / eigvals.sum())
    return np.array(curvature)

def hausdorff(p1, p2):
    return max(directed_hausdorff(p1, p2)[0], directed_hausdorff(p2, p1)[0])

# -------------------------
# Main loop
# -------------------------
results = []

for pid in PATIENT_IDS:
    print(f"Processing {pid}...")
    try:
        mesh = load_mesh(pid)
        worn = simulate_wear_trimesh(mesh)

        pts1 = sample_points(mesh, 1000)
        pts2 = sample_points(worn, 1000)

        curv1 = estimate_curvature(pts1)
        curv2 = estimate_curvature(pts2)
        d_curv = np.abs(curv1 - curv2)

        haus = hausdorff(pts1, pts2)

        results.append({
            "PatientID": pid,
            "MeanCurvΔ": np.mean(d_curv),
            "MaxCurvΔ": np.max(d_curv),
            "Hausdorff": haus
        })

    except Exception as e:
        print(f"Error with {pid}: {e}")

# -------------------------
# Save results
# -------------------------
df = pd.DataFrame(results)
df.to_csv("multi_patient_wear_results.csv", index=False)
print("✅ All done! Results saved.")
