import os, glob, cv2, json, time, psutil
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.svm import train_svm
from src.rf import train_rf
from src.knn import train_knn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(dataset_path, img_size=(64, 64)):
    X, y = [], []
    classes = sorted(os.listdir(dataset_path))
    print(f"[HOG] Classes: {classes}")
    for label in classes:
        for f in glob.glob(os.path.join(dataset_path, label, "*.jpg")):
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            X.append(cv2.resize(img, img_size))
            y.append(label)
    print(f"[HOG] Loaded {len(X)} images")
    return np.array(X), np.array(y)


def extract_hog_features(
    images,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    orientations=9,
    dtype=np.float32,
    memmap_path=None,
):
    n = len(images)
    length = hog(
        images[0],
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        transform_sqrt=True,
    ).shape[0]
    feats = (
        np.memmap(memmap_path, mode="w+", dtype=dtype, shape=(n, length))
        if memmap_path
        else np.zeros((n, length), dtype=dtype)
    )
    for i, img in enumerate(images):
        if i % 1000 == 0 and i > 0:
            print(f"[HOG] {i}/{n} processed")
        feats[i] = hog(
            img,
            orientations,
            pixels_per_cell,
            cells_per_block,
            block_norm="L2-Hys",
            transform_sqrt=True,
        ).astype(dtype)
    print("[HOG] Feature extraction complete.")
    return feats


def hog_function(
    dataset_path,
    test_size=0.2,
    random_state=42,
    memmap_path="Features/hog_feats.dat",
    output_path="Results/hog_results.json",
):
    print("[HOG] Pipeline start")
    images, labels = load_data(dataset_path)
    feats = extract_hog_features(images, memmap_path=memmap_path)
    X_tr, X_te, y_tr, y_te = train_test_split(
        feats, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    scaler = StandardScaler()
    X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)

    # Reduce to exactly 128 dimensions
    pca = PCA(n_components=128, random_state=42)
    X_tr_p, X_te_p = pca.fit_transform(X_tr_s), pca.transform(X_te_s)

    # Train models
    svm_m = train_svm(X_tr_p, y_tr)
    rf_m = train_rf(X_tr, y_tr)
    knn_m = train_knn(X_tr_p, y_tr)

    # Evaluate and collect metrics
    results = {}
    proc = psutil.Process()
    for name, mdl, X_e in [
        ("svm", svm_m, X_te_p),
        ("rf", rf_m, X_te),
        ("knn", knn_m, X_te_p),
    ]:
        print(f"[HOG] Evaluating {name}")
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
        plt.title(f"HOG {name} Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.savefig(f"Results/hog_{name}_confusion_matrix.png")  # Save the figure

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
    print(f"[HOG] Results saved to {output_path}")
    return results
