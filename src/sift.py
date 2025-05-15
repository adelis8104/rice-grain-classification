import os, glob, cv2, json, time, psutil
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.svm import train_svm
from src.rf import train_rf
from src.knn import train_knn
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import seaborn as sns


def load_sift_data(dataset_path, img_size=(64, 64)):
    X, y = [], []
    classes = sorted(os.listdir(dataset_path))
    print(f"[SIFT] Classes: {classes}")
    for label in classes:
        for f in glob.glob(os.path.join(dataset_path, label, "*.jpg")):
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            X.append(cv2.resize(img, img_size))
            y.append(label)
    print(f"[SIFT] Loaded {len(X)} images")
    return np.array(X), np.array(y)


def extract_sift_features(images, dtype=np.float32, memmap_path=None, n_clusters=128):
    sift = cv2.SIFT_create()
    all_descs = []
    for img in images:
        _, des = sift.detectAndCompute(img, None)
        if des is not None:
            all_descs.append(des)
    if not all_descs:
        raise ValueError("No SIFT descriptors found in any image.")
    all_descs = np.vstack(all_descs)
    print(
        f"[SIFT] Clustering {all_descs.shape[0]} descriptors into {n_clusters} words..."
    )

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=min(10000, all_descs.shape[0]),
        random_state=42,
        n_init=3,
    )
    kmeans.fit(all_descs)

    n = len(images)
    K = n_clusters
    if memmap_path:
        os.makedirs(os.path.dirname(memmap_path), exist_ok=True)
        feats = np.memmap(memmap_path, mode="w+", dtype=dtype, shape=(n, K))
    else:
        feats = np.zeros((n, K), dtype=dtype)

    for i, img in enumerate(images):
        _, des = sift.detectAndCompute(img, None)
        if des is not None:
            words = kmeans.predict(des)
            hist, _ = np.histogram(words, bins=np.arange(K + 1), density=True)
            feats[i] = hist.astype(dtype)

        if i > 0 and i % 1000 == 0:
            print(f"[SIFT] {i}/{n} processed")
    print("[SIFT] feature extraction complete.")

    return feats


def sift_function(
    dataset_path,
    test_size=0.2,
    random_state=42,
    memmap_path="Features/sift_feats.dat",
    output_path="Results/sift_results.json",
):
    print("[SIFT] Pipeline start")
    images, labels = load_sift_data(dataset_path)
    feats = extract_sift_features(images, memmap_path=memmap_path)
    X_tr, X_te, y_tr, y_te = train_test_split(
        feats, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    scaler = StandardScaler()
    X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)

    # Train models
    svm_m = train_svm(X_tr_s, y_tr)
    rf_m = train_rf(X_tr, y_tr)
    knn_m = train_knn(X_tr_s, y_tr)

    # Evaluate and collect metrics
    results = {}
    proc = psutil.Process()
    for name, mdl, X_e in [
        ("svm", svm_m, X_te_s),
        ("rf", rf_m, X_te),
        ("knn", knn_m, X_te_s),
    ]:
        print(f"[SIFT] Evaluating {name}")
        mem_before, t0 = proc.memory_info().rss, time.perf_counter()
        y_p = mdl.predict(X_e)
        t1, mem_after = time.perf_counter(), proc.memory_info().rss

        cm = confusion_matrix(y_te, y_p)
        class_labels = sorted(np.unique(y_te))

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels,
        )
        plt.title(f"SIFT {name} Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.savefig(f"Results/sift_{name}_confusion_matrix.png")  # Save the figure

        results[name] = {
            "best_params": getattr(mdl, "best_params_", {}),
            "accuracy": accuracy_score(y_te, y_p),
            "inference_time": t1 - t0,
            "memory_usage": mem_after - mem_before,
            "report": classification_report(y_te, y_p, output_dict=True),
            "conf_mat": cm.tolist(),
        }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SIFT] Results saved to {output_path}")
    return results
