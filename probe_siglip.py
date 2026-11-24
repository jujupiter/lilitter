"""Linear probe on SigLIP: keep its image encoder frozen and train only a 6-class head on trashnet + RealWaste + TACO.

Zero-shot SigLIP beats every fully trained model on the Commons photos (#5): training the whole network overwrites the
general knowledge that unfamiliar photos need. A head on frozen features keeps that knowledge, and costs one pass of
the encoder over the images plus a small logistic regression.

1. Encode every training, validation and test image once. The features are cached in models/features_<arch>.pt, and
   a later run only encodes the images that are not cached yet.
2. Fit a logistic regression on the training features for each regularisation strength C; keep the one with the best
   mean validation accuracy over the three sources (as train_mixed.py does). Each source weighs in the loss as in
   train_mixed.py's batches.
3. With --zero-shot-weight W, blend the probe with the zero-shot classifier of zero_shot.py, which is a linear head
   on the same features: head = (1 - W) * probe + W * zero-shot. Both give scores on a similar scale (checked on the
   validation set), so W = 0.5 weighs them equally. The blend is one linear head again: no extra cost to run it.
   With --zero-shot-weight dev, W is chosen on the development set (data/commons_dev/, #40) among DEV_WEIGHTS: the
   best mean accuracy over its studio and outdoor photos. The development set is never a test set.
4. Save the encoder with its new head to models/<arch>_probe.pth, which benchmark.py and predict.py can load, and
   score every test set from the cached features: results/benchmark_<arch>_probe.csv, in benchmark.py's format.

usage: python probe_siglip.py [--zero-shot-weight 0.5 | dev] [--limit N]
--zero-shot-weight W: outputs end in _zs<W x 100>, e.g. models/siglip_base_probe_zs50.pth; with `dev`, in _zsdev.
W is never tuned on the test sets; the validation sets come from the training sources and would always favour W = 0.
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
from common import (ARCHS, CLASSES, CLASS_TO_IDX, DEVICE, NUM_WORKERS, TrashDataset, build_model, load_splits, measured_run,
                    save_model)
from train_mixed import BATCH_MIX
from zero_shot import SCALE, text_head

ARCH = "siglip_base"
C_GRID = [1, 3, 10, 30, 100, 300, 1000]  # features have length 1, so the head needs little regularisation
DEV = Path("data/commons_dev/sources.csv")
DEV_WEIGHTS = [0, 0.25, 0.33, 0.5, 0.67, 0.75, 1]
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
    parser.add_argument("--zero-shot-weight", default="0",
                        help="weight of the zero-shot head in the blend, from 0 (probe only) to 1 (zero-shot only), "
                             "or `dev` to choose it on the development set")
    parser.add_argument("--limit", type=int, help="only N images per source and split, to check the script runs")
    args = parser.parse_args()
    choose = args.zero_shot_weight == "dev"
    w = None if choose else float(args.zero_shot_weight)
    suffix = ("_zsdev" if choose else f"_zs{round(w * 100)}" if w else "") + (f"_limit{args.limit}" if args.limit else "")
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
    dev = pd.DataFrame(columns=["path", "label", "setting"])
    if choose:
        dev = pd.read_csv(DEV).dropna(subset=["label"])  # photos with a J-code only wait for phase 2, as in benchmark.py
        dev = dev[dev.path.map(lambda p: Path(p).exists())].reset_index(drop=True)
        print("Development set:", dev.groupby("setting").size().to_dict())

    with measured_run("probe_siglip.py", out):
        model = build_model(ARCH).eval()  # pretrained encoder; its head is replaced below
        out.parent.mkdir(exist_ok=True)
        feats = encode(model, train.path.tolist() + val.path.tolist() + bench.path.tolist() + dev.path.tolist(),
                       Path(f"models/features_{ARCH}.pt")).numpy()
        x_train, x_val, x_bench, x_dev = np.split(feats, np.cumsum([len(train), len(val), len(bench)]))
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

        weight, bias = torch.from_numpy(clf.coef_).float(), torch.from_numpy(clf.intercept_).float()
        if choose or w:
            from transformers import AutoModel, AutoProcessor
            name = ARCHS[ARCH]
            clip = AutoModel.from_pretrained(name, use_safetensors=True).eval()
            zero_shot = SCALE * text_head(clip, AutoProcessor.from_pretrained(name)).cpu()
            del clip
        if choose:
            best = None
            y_dev = dev.label.map(CLASS_TO_IDX).to_numpy()
            for v in DEV_WEIGHTS:
                scores = torch.from_numpy(x_dev) @ ((1 - v) * weight + v * zero_shot).T + (1 - v) * bias
                acc = dev.assign(ok=scores.argmax(1).numpy() == y_dev).groupby("setting").ok.mean()
                print(f"Zero-shot weight {v:<4} | dev acc " + " ".join(f"{s} {a:.3f}" for s, a in acc.items())
                      + f" | mean {acc.mean():.3f}")
                if best is None or acc.mean() > best[0]:
                    best = (acc.mean(), v)
            w = best[1]
            print(f"Chosen on the development set: zero-shot weight {w} (mean dev acc {best[0]:.4f})")
        if w:
            weight, bias = (1 - w) * weight + w * zero_shot, (1 - w) * bias
            scores = torch.from_numpy(x_val) @ weight.T + bias
            acc = val.assign(ok=scores.argmax(1).numpy() == val.label.map(CLASS_TO_IDX)).groupby("source").ok.mean()
            print(f"Blend, zero-shot weight {w} | val acc " + " ".join(f"{s} {a:.3f}" for s, a in acc.items())
                  + f" | mean {acc.mean():.3f}")

        with torch.no_grad():
            model.head.weight.copy_(weight)
            model.head.bias.copy_(bias)
            # the saved model must give what the cached features give
            images, _ = next(iter(DataLoader(TrashDataset(bench.head(4), model.transform), batch_size=4)))
            direct = model(images.to(DEVICE)).softmax(1).cpu().numpy()
        probs = (torch.from_numpy(x_bench) @ weight.T + bias).softmax(1).numpy()
        assert np.allclose(direct, probs[:len(direct)], atol=1e-3), "the saved model disagrees with the cached features"
        save_model(model, ARCH, out)

    res = bench.assign(pred=[CLASSES[i] for i in probs.argmax(1)], conf=probs.max(1),
                       ok=lambda d: d.pred == d.label)
    res.to_csv(out_csv, index=False)
    print(f"Saved {out} and {out_csv}: `python benchmark.py {out_csv}` prints the full report")
    print(res.groupby("benchmark", sort=False).ok.mean().round(3).to_string())


if __name__ == "__main__":
    main()
