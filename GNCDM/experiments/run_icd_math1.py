"""Run EduCDM ICD on math1 (CPU) with the authors' official hyper-params.

Official example (examples/ICD/ICD.py) main() defaults: alpha=0.2, beta=0.9,
tolerance=0.2, epoch=1, warmup_ratio=0.1, weight_decay=0 (ncd), inner_metrics=True,
prequential eval (stream i+1 tests model trained up to i). The example's "math" entry
is a DIFFERENT dataset (10269x17747x1488) — we keep math1 dims and only borrow hparams.

We append math1 test as the final stream so the last eval is a held-out test number,
while inner_metrics=True also yields the prequential train-stream metrics.

Usage: run_icd_math1.py [stream_num=50] [cdm=ncd] [alpha=0.2] [tolerance=0.2] [epoch=1]
"""

import json
import logging
import os
import sys

from EduCDM.ICD.ICD import ICD
from EduCDM.ICD.etl import extract, inc_stream

HERE = os.path.dirname(os.path.abspath(__file__))
ICD_DATA = os.path.join(HERE, "icd_math1")
OUT = os.path.join(HERE, "icd_out")
os.makedirs(OUT, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("icd_math1")

USER_N, ITEM_N, KNOW_N = 4209, 20, 11  # math1 dims (CLAUDE.md)

# official main() hyper-params (overridable via argv)
STREAM_NUM = int(sys.argv[1]) if len(sys.argv) > 1 else 50
CDM = sys.argv[2] if len(sys.argv) > 2 else "ncd"
ALPHA = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
TOLERANCE = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
EPOCH = int(sys.argv[5]) if len(sys.argv) > 5 else 1
BETA, WARMUP, WD = 0.9, 0.1, (1e-4 if CDM == "mirt" else 0)

item_csv = os.path.join(ICD_DATA, "item.csv")
train_df, _, _, i2k = extract(os.path.join(ICD_DATA, "train.csv"), item_csv)
test_df, _, _, _ = extract(os.path.join(ICD_DATA, "test.csv"), item_csv)

stream_size = int(len(train_df) // STREAM_NUM)
train_chunks = list(inc_stream(train_df, stream_size))
inc_list = train_chunks + [test_df]  # final chunk = held-out test eval
logger.info(
    "cdm=%s alpha=%s tol=%s epoch=%d beta=%s warmup=%s | train=%d -> %d chunks (~%d rows) + test(%d)",
    CDM,
    ALPHA,
    TOLERANCE,
    EPOCH,
    BETA,
    WARMUP,
    len(train_df),
    len(train_chunks),
    stream_size,
    len(test_df),
)

model = ICD(
    CDM,
    USER_N,
    ITEM_N,
    KNOW_N,
    epoch=EPOCH,
    weight_decay=WD,
    inner_metrics=True,
    logger=logger,
    alpha=ALPHA,
    ctx="cpu",
)

os.chdir(OUT)  # contain baize checkpoint artifacts
wfs = {
    h: open(os.path.join(OUT, f"{h}.jsonl"), "w") for h in ("metrics", "trait", "inc_trait", "tp")
}
model.train(inc_list, i2k, beta=BETA, warmup_ratio=WARMUP, tolerance=TOLERANCE, wfs=wfs)
for f in wfs.values():
    f.close()

lines = [
    json.loads(ln) for ln in open(os.path.join(OUT, "metrics.jsonl")).read().strip().splitlines()
]
assert lines, "no metrics captured"
accs = [r["metrics"].get("accuracy") for r in lines if "accuracy" in r["metrics"]]
aucs = [r["metrics"].get("macro_auc") for r in lines if "macro_auc" in r["metrics"]]
print("\n===== ICD (cdm=%s) on math1 — %d eval points =====" % (CDM, len(lines)))
if accs:
    print("prequential acc:  mean=%.4f  last(=test)=%.4f" % (sum(accs) / len(accs), accs[-1]))
    print("prequential auc:  mean=%.4f  last(=test)=%.4f" % (sum(aucs) / len(aucs), aucs[-1]))
print("final test eval:", json.dumps(lines[-1]["metrics"], ensure_ascii=False))
