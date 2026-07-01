"""Convert math1 Q_matrix.npy -> ICD item.csv (item_id, knowledge_code).

ICD's etl.item2knowledge does: eval(knowledge_code) then `k - k_offset` (k_offset=1),
so knowledge_code must hold 1-indexed concept ids. math1 Q is (20, 11), concepts 0..10
-> write k+1. log csvs already match ICD (user_id,item_id,score), no conversion needed.
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "GNCDM", "data")
OUT = os.path.join(HERE, "icd_math1")
os.makedirs(OUT, exist_ok=True)

Q = np.load(os.path.join(DATA, "math1_Q_matrix.npy"))  # (20, 11) int
rows = []
for item_id in range(Q.shape[0]):
    ks = [int(k) + 1 for k in np.where(Q[item_id] > 0)[0]]  # 1-indexed
    rows.append({"item_id": item_id, "knowledge_code": str(ks)})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "item.csv"), index=False)

# logs: ICD wants a single stream-able log; reuse train as incremental train, test for eval.
for split in ("train", "valid", "test"):
    src = os.path.join(DATA, f"math1_{split}_0.8_0.2.csv")
    pd.read_csv(src).to_csv(os.path.join(OUT, f"{split}.csv"), index=False)

print(f"wrote {OUT}/item.csv ({Q.shape[0]} items) + train/valid/test.csv")
print(pd.read_csv(os.path.join(OUT, "item.csv")).head().to_string(index=False))


# ponytail: self-check — every item has >=1 concept, ids within 1..n_know
def _check():
    df = pd.read_csv(os.path.join(OUT, "item.csv"))
    assert len(df) == Q.shape[0]
    for code in df["knowledge_code"]:
        ks = eval(code)
        assert ks and all(1 <= k <= Q.shape[1] for k in ks), code
    print("self-check OK")


if __name__ == "__main__":
    _check()
