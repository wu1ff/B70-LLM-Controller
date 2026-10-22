"""Read saved real-worker tensors and compare all four TP replicas."""
import json
from pathlib import Path
import torch

folder=Path('/work/feature-audit')
results=[]
for capture in range(2):
    samples=[torch.load(folder/f'rank-{rank}-capture-{capture}.pt',weights_only=True) for rank in range(4)]
    ref=samples[0]
    for rank,sample in enumerate(samples):
        assert sample['dispatch']['boundary_ids']==[6,16,26,36,46]
        assert sample['dispatch']['stage_bytes']==1310720
        assert sample['dispatch']['table_weight_elements']==0
        assert sample['dispatch']['table_pinned'] is False
        assert sample['dispatch']['residency_shim_mapped']
        assert sample['dispatch']['gdn_library_mapped']
        assert sample['dummy_profile_calls_before']>0
        assert sample['disk_generation']>sample['dummy_profile_calls_before']
        for got,want in zip(sample['features'],sample['reference']):
            torch.testing.assert_close(got,want,atol=.0078125,rtol=.008)
        torch.testing.assert_close(sample['concatenated'],torch.cat(sample['features'],-1),rtol=0,atol=0)
        torch.testing.assert_close(sample['positions'],ref['positions'],rtol=0,atol=0)
        torch.testing.assert_close(sample['concatenated'],ref['concatenated'],rtol=0,atol=0)
        if sample['input_ids'] is not None:
            torch.testing.assert_close(sample['input_ids'],ref['input_ids'],rtol=0,atol=0)
        results.append({'rank':rank,'capture':capture,'shape':list(sample['concatenated'].shape),
                        'max_abs_errors':sample['max_abs_errors'],'dummy_profile_calls':sample['dummy_profile_calls_before'],
                        'tp4_bitwise_equal':True,'passed':True})
Path('/work/live-feature-results.json').write_text(json.dumps({'passed':True,'results':results},indent=2)+'\n')
print(json.dumps({'passed':True,'results':results},indent=2))
