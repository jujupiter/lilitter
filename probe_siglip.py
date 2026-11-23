"""Linear probe on SigLIP: keep its image encoder frozen and train only a 6-class head on trashnet + RealWaste + TACO.

Zero-shot SigLIP beats every fully trained model on the Commons photos (#5): training the whole network overwrites the
general knowledge that unfamiliar photos need. A head on frozen features keeps that knowledge, and costs one pass of
the encoder over the images plus a small logistic regression.

1. Encode every training, validation and test image once. The features are cached in models/features_<arch>.pt, and
   a later run only encodes the images that are not cached yet.
2. Fit a logistic regression on the training features for each regularisation strength C; keep the one with the best
   mean validation accuracy over the three sources (as train_mixed.py does). Each source weighs in the loss as in
   train_mixed.py's batches.
3. Save the encoder with its new head to models/<arch>_probe.pth, which benchmark.py and predict.py can load, and
   score every test set from the cached features: results/benchmark_<arch>_probe.csv, in benchmark.py's format.

usage: python probe_siglip.py [--limit N]
--limit N uses only N images per source and split, to check that the script runs; its outputs end in _limitN.
The run's energy goes to results/runs.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from benchmark import benchmarks
from common import (CLASSES, CLASS_TO_IDX, DEVICE, NUM_WORKERS, TrashDataset, build_model, load_splits, measured_run,
                    save_model)
from train_mixed import BATCH_MIX

ARCH = "siglip_base"
C_GRID = [1, 3, 10, 30, 100, 300, 1000]  # features have length 1, so the head needs little regularisation
BATCH = 32
SAVE_EVERY = 20  # batches: an interrupted run keeps most of its work


def encode(model, paths, cache_path):
    """Features of `paths` (a tensor, one row per path), encoding only those not in the cache."""
    cache = torch.load(cache_path, weights_only=True) if cache_path.exists() else {"paths": [], "features": torch.empty(0)}
    known = dict(zip(cache["paths"], cache["features"]))
    todo = [p for p in dict.fromkeys(paths) if p not in known]
    print(f"Features: {len(paths) - len(todo)} cached, {len(todo)} to encode")
    if todo:
        df = pd.DataFrame({"path": todo, "label": CLASSES[0]})  # TrashDataset wants a label; it is not used
        loader = DataLoader(TrashDataset(df, model.transform), batch_size=BATCH, num_workers=NUM_WORKERS)
        done = 0
        with torch.no_grad():
            for step, (images, _) in enumerate(loader, 1):
                feats = model.features(images.to(DEVICE)).float().cpu()
                known.update(zip(todo[done:done + len(feats)], feats))
                done += len(feats)
                if step % SAVE_EVERY == 0 or done == len(todo):
                    torch.save({"paths": list(known), "features": torch.stack(list(known.values()))}, cache_path)
                    print(f"  encoded {done}/{len(todo)}", flush=True)
    return torch.stack([known[p] for p in paths])


def source_weights(sources):
    """Each source weighs in the loss as its share of a train_mixed.py batch, whatever its number of images."""
    counts = sources.value_counts()
    share = {s: k / sum(BATCH_MIX.values()) for s, k in BATCH_MIX.items()}
    return sources.map(lambda s: share[s] / counts[s]).to_numpy() * len(sources)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="only N images per source and split, to check the script runs")
    args = parser.parse_args()
    suffix = f"_limit{args.limit}" if args.limit else ""
    out = Path(f"models/{ARCH}_probe{suffix}.pth")
    out_csv = Path(f"results/benchmark_{ARCH}_probe{suffix}.csv")

    df = load_splits()
    bench = benchmarks()
    if args.limit:
        df = df.groupby(["source", "split"]).head(args.limit)
        bench = bench.groupby("benchmark").head(args.limit)
    train = df[df.split == "train"].reset_index(drop=True)
    val = df[df.split == "val"].reset_index(drop=True)
    bench = bench.reset_index(drop=True)

    with measured_run("probe_siglip.py", out):
        model = build_model(ARCH).eval()  # pretrained encoder; its head is replaced below
        out.parent.mkdir(exist_ok=True)
        feats = encode(model, train.path.tolist() + val.path.tolist() + bench.path.tolist(),
                       Path(f"models/features_{ARCH}.pt")).numpy()
        x_train, x_val, x_bench = np.split(feats, [len(train), len(train) + len(val)])
        y_train = train.label.map(CLASS_TO_IDX).to_numpy()
        weights = source_weights(train.source)

        best = None
        for c in C_GRID:
            clf = LogisticRegression(C=c, max_iter=5000).fit(x_train, y_train, sample_weight=weights)
            acc = val.assign(ok=clf.predict(x_val) == val.label.map(CLASS_TO_IDX)).groupby("source").ok.mean()
            print(f"C {c:>5} | val acc " + " ".join(f"{s} {a:.3f}" for s, a in acc.items()) + f" | mean {acc.mean():.3f}")
            if best is None or acc.mean() > best[0]:
                best = (acc.mean(), c, clf)
        _, c, clf = best
        print(f"Best: C {c}, mean val acc {best[0]:.4f}")

        with torch.no_grad():
            model.head.weight.copy_(torch.from_numpy(clf.coef_).float())
            model.head.bias.copy_(torch.from_numpy(clf.intercept_).float())
            # the saved model must give what the cached features give
            images, _ = next(iter(DataLoader(TrashDataset(bench.head(4), model.transform), batch_size=4)))
            direct = model(images.to(DEVICE)).softmax(1).cpu().numpy()
        probs = clf.predict_proba(x_bench)
        assert np.allclose(direct, probs[:len(direct)], atol=1e-3), "the saved model disagrees with the cached features"
        save_model(model, ARCH, out)

    res = bench.assign(pred=[CLASSES[i] for i in probs.argmax(1)], conf=probs.max(1),
                       ok=lambda d: d.pred == d.label)
    res.to_csv(out_csv, index=False)
    print(f"Saved {out} and {out_csv}: `python benchmark.py {out_csv}` prints the full report")
    print(res.groupby("benchmark", sort=False).ok.mean().round(3).to_string())


if __name__ == "__main__":
    main()
