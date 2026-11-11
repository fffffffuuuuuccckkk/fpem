"""CPU smoke and causal safety checks for the G1/R3 mechanism suite."""
import unittest

import torch

from tools.fpem_abc_mechanism import nearest, synthetic_g1, synthetic_r3
from tools.fpem_abc_probability import ConditionedResidualFlow, DIM, packet


class MechanismTests(unittest.TestCase):
    def test_causal_neighbours_have_completed_future(self):
        source=torch.randn(5,1,64)
        query=source[2:4]
        ids,_,valid=nearest(source,query,k=3,
            source_ids=torch.tensor([0,5,10,15,20]),
            query_ids=torch.tensor([12,18]),span=10)
        self.assertTrue(torch.all(torch.tensor([0,5,10,15,20])[ids][valid]+10 <=
                                  torch.tensor([12,18])[:,None].expand_as(ids)[valid]))

    def test_synthetic_shared_and_local_truth(self):
        self.assertGreater(synthetic_g1()["shared_map_cosine"],.8)
        self.assertGreater(synthetic_r3()["route_accuracy"],.5)

    def test_condition_capacity_equal_and_no_future_target_input(self):
        model=ConditionedResidualFlow(24)
        record={"future":torch.randn(2,3,64),"y_inv":torch.randn(2,24,3),
                "truth":torch.randn(2,24,3),"scale":torch.ones(2,1,3),
                "features":torch.randn(2,3,DIM),"ids":torch.arange(2)}
        base=packet(record,torch.arange(2),"cpu","base_condition")
        full=packet(record,torch.arange(2),"cpu","all_fused")
        self.assertEqual(base[0].shape,(2,3,2,64))
        self.assertEqual(full[-1].shape,(2,3,2,DIM))
        self.assertTrue(torch.equal(model.extra.weight,torch.zeros_like(model.extra.weight)))
        self.assertTrue(torch.equal(base[-1],torch.zeros_like(base[-1])))
        self.assertFalse(torch.equal(full[-1],torch.zeros_like(full[-1])))


if __name__=="__main__":
    unittest.main()
