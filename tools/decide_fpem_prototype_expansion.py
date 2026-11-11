"""Gate full 84-case expansion on actual pilot VAL and TRAIN forward evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True)
    p.add_argument("--backbone",required=True)
    p.add_argument("--dataset",required=True)
    cli=p.parse_args()
    root=Path(cli.root)
    evidence=[]
    for horizon in (96,720):
        case=root/cli.backbone/cli.dataset/f"pred_{horizon}"
        p01=load(case/"shared/p01/metrics.json")
        oof=load(case/"forward/metrics.json")
        candidates=[]
        for name in ("risk_var","risk_dro","shared_grad_norm","shared_grad_variance",
                     "shared_irm","conditional","subspace","ref_prototype",
                     "map_consensus","map_spectral","pairwise","shared_private",
                     "shared_private_irm","adversarial_weak","adversarial_strong",
                     "map_projection","map_shared_irm"):
            result=load(case/"shared"/name/"metrics.json")
            if result is not None:candidates.append((result["val"]["mse"],name))
        if p01 is None or oof is None or not candidates:
            evidence.append({"horizon":horizon,"eligible":False,"reason":"missing pilot evidence"})
            continue
        best=min(candidates)
        # The VAL and chronological evidence must concern the SAME method.
        # A mapping method's low VAL score cannot borrow a direct head's OOF gain.
        forward_method=oof["method"]
        matched=load(case/"shared"/forward_method/"metrics.json")
        eligible=(matched is not None and forward_method not in ("p01","p02")
                  and matched["val"]["mse"]<p01["val"]["mse"]
                  and oof["shared"]["mse"]<oof["a0"]["mse"])
        evidence.append({"horizon":horizon,"eligible":eligible,
                         "best_method":forward_method,"best_val_method":best[1],
                         "matched_prototype_val":matched["val"]["mse"] if matched else None,
                         "best_prototype_val":best[0],"p01_val":p01["val"]["mse"],
                         "forward_shared":oof["shared"]["mse"],"forward_a0":oof["a0"]["mse"]})
    # One confirmed horizon is sufficient to study transfer; no unearned
    # automatic claim that the method works on the other horizon.
    expand=any(row["eligible"] for row in evidence)
    chosen=min((row for row in evidence if row["eligible"]),
               key=lambda row:row["matched_prototype_val"],default=None)
    payload={"expand":expand,"selected_method":chosen["best_method"] if chosen else None,
             "selected_horizon":chosen["horizon"] if chosen else None,
             "evidence":evidence,
             "rule":"prototype method beats P01 on VAL and shared head beats A0 in TRAIN forward-OOF"}
    (root/cli.backbone/"expansion_decision.json").write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload))


if __name__=="__main__":main()
