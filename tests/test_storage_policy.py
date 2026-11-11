"""Tiny, isolated storage safety tests; never touch real results."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from tools.storage.audit_results import audit
from tools.storage.cleanup_results import cleanup
from tools.storage.common import active_file,sha256,stat_fingerprint
from tools.storage.prune_run_artifacts import prune
from tools.storage.retention import promote_if_best,record_trial,require_free_space


class StoragePolicyTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="fpem_storage_test_")
        self.root=Path(self.temp.name);self.results=self.root/"results";self.results.mkdir()

    def tearDown(self):self.temp.cleanup()

    def test_audit_blocks_partial_recent_cache_and_symlink(self):
        run=self.results/"fpem_eiil3_20261110"/"Electricity";run.mkdir(parents=True)
        cache=run/"features.npz";cache.write_bytes(b"placeholder")
        link=self.results/"outside_link";link.symlink_to(cache)
        audit(self.results,self.root/"report")
        self.assertEqual((self.root/"report"/"candidates.jsonl").read_text(),"")
        rows=(self.root/"report"/"inventory.csv").read_text()
        self.assertIn("unfinished EIIL3",rows)
        self.assertIn("symbolic link",rows)

    def test_incomplete_and_inode_changed_never_deleted(self):
        run=self.results/"run";run.mkdir();target=run/"frozen_a0_cache.pt"
        target.write_bytes(b"data")
        proof=self.root/"proof.json";proof.write_text(json.dumps({"status":"passed",
            "cache_path":str(target),"fingerprint":stat_fingerprint(target)}))
        item={"kind":"REGENERABLE_CACHE","path":str(target),
              "fingerprint":stat_fingerprint(target),"verification":str(proof),
              "expected_sha256":sha256(target)}
        manifest=self.root/"candidate.jsonl";manifest.write_text(json.dumps(item)+"\n")
        result=cleanup(manifest,self.results,apply=True,recent_hours=0)
        self.assertEqual(result[0]["status"],"BLOCKED")
        self.assertTrue(target.exists())
        target.write_bytes(b"changed")
        result=cleanup(manifest,self.results,apply=True,recent_hours=0)
        self.assertIn("stat changed",result[0]["block_reason"])

    def test_active_file_and_storage_floor(self):
        target=self.results/"open.bin";target.write_bytes(b"x")
        with target.open("rb"):
            self.assertIsNotNone(active_file(target))
        with self.assertRaises(OSError):require_free_space(self.results,min_free_gib=10**9)

    def test_best_replacement_and_formal_ablation_protection(self):
        run=self.results/"trick";run.mkdir()
        policy=self.root/"policy.yaml";policy.write_text(
            "trick_search:\n  best_checkpoints_per_comparison: 1\n")
        source1=self.root/"first.pt";source2=self.root/"second.pt"
        torch.save({"w":torch.tensor([1.])},source1)
        torch.save({"w":torch.tensor([2.])},source2)
        first={"trial_id":"a","params":{},"seed":1,"epoch":1,"status":"complete",
               "objective":"VAL_MSE","selected_by":"VAL","comparison_key":"case/seed1",
               "VAL_MSE":0.5}
        second={**first,"trial_id":"b","VAL_MSE":0.4}
        record_trial(run,first);self.assertTrue(promote_if_best(run,first,source1,"case/seed1","VAL_MSE"))
        old_hash=sha256(run/"best_model.pt")
        invalid=self.root/"invalid.pt";invalid.write_bytes(b"not a model")
        with self.assertRaises(Exception):promote_if_best(run,second,invalid,"case/seed1","VAL_MSE")
        self.assertEqual(sha256(run/"best_model.pt"),old_hash)
        record_trial(run,second)
        self.assertTrue(promote_if_best(run,second,source2,"case/seed1","VAL_MSE"))
        self.assertEqual(len((run/"trials.jsonl").read_text().splitlines()),2)
        self.assertEqual(len(list(run.glob("best_model.pt"))),1)
        scratch=run/"scratch";scratch.mkdir();(scratch/"loser.pt").write_bytes(b"loser")
        (run/"complete").write_text("yes")
        self.assertEqual(len(prune(run,policy,dry_run=True)),1)
        prune(run,policy,dry_run=False)
        self.assertFalse(scratch.exists())
        (run/"formal_ablation").write_text("protected")
        with self.assertRaises(ValueError):prune(run,policy,dry_run=False)

    def test_eiil3_nonpersistent_feature_smoke_matches_legacy_cache(self):
        import tools.run_fpem_eiil3 as eiil
        record={"design":np.zeros((4,2),np.float32),"key":np.ones((4,2),np.float32),
                "y":np.ones((4,1),np.float32),"a0":np.zeros((4,1),np.float32),
                "ratio":np.ones((4,1),np.float32),"start":np.arange(4),
                "channel":np.zeros(4,np.int64)}
        raw={**record,"ids":torch.arange(4),"x":torch.arange(8,dtype=torch.float32).reshape(4,2,1)}
        model=SimpleNamespace(h_forecast_head=SimpleNamespace(
            weight=torch.zeros((1,2)),bias=torch.zeros(1)))
        dataset=SimpleNamespace(dataset=SimpleNamespace(dataset="unused"))
        loaders={space:dataset for space in ("train","val","test")}
        q=np.tile(np.array([[.2,.3,.5]],np.float32),(4,1))
        args=SimpleNamespace(seq_len=2,pred_len=1,dataset_name="synthetic",backbone="PatchTST")
        def source(cache,name):
            output=self.root/name;output.mkdir()
            (output/"teacher_q.npz").write_bytes(b"small teacher")
            return SimpleNamespace(output=output,asset={"A0_sha256":"fixed",
                "base_run_config":"unused"},gpu=0,seed=1,train_stride=1,
                val_stride=1,test_stride=1,persist_features=cache)
        with patch.object(eiil,"load_experiment",return_value=(args,model,loaders,"cpu","fixed")), \
             patch.object(eiil,"collect",return_value=raw), \
             patch.object(eiil,"prepare",side_effect=lambda value,floor:value), \
             patch.object(eiil,"flat_record",side_effect=lambda value:record), \
             patch.object(eiil,"split_train",return_value=({k:record for k in eiil.SPACES[:4]},[1,2,3])), \
             patch.object(eiil,"valid_teacher",return_value=(q,{"source":"synthetic TRAIN"})):
            sparse=source(False,"nonpersistent");dense=source(True,"persistent")
            d0,m0,h0,q0=eiil.get_data(sparse)
            d1,m1,h1,q1=eiil.get_data(dense)
        self.assertFalse((sparse.output/"features.npz").exists())
        self.assertTrue((dense.output/"features.npz").exists())
        for space in eiil.SPACES:
            for field in record:np.testing.assert_array_equal(d0[space][field],d1[space][field])
        for space in eiil.SPACES[:4]:np.testing.assert_array_equal(q0[space],q1[space])
        self.assertEqual(m0["init_error_max"],0.0)
        self.assertEqual(m1["init_error_max"],0.0)
        d2,m2,h2,q2=eiil.get_data(dense)
        np.testing.assert_array_equal(d2["test"]["a0"],d0["test"]["a0"])
        np.testing.assert_array_equal(q2["A"],q0["A"])


if __name__=="__main__":unittest.main()
