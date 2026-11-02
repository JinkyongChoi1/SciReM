"""Diagnostic additions cannot invalidate or silently alter a reused primary result."""
import json
import tempfile
import unittest
from pathlib import Path
from analysis_reuse import VerifiedAnalysisCache
from four_role_resume import verify_reused_result


class DiagnosticCacheResumeTests(unittest.TestCase):
    def test_only_receipt_bound_additions_are_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); first=root/'attempt-0001'; first.mkdir()
            cache=VerifiedAnalysisCache({'sha256':'data'})
            action={'family':'logit','terms':['x']}
            source={'returncode':0,'effective_action':action,'estimates':[{'coefficient':'.1'}],'metrics':[{'n':'10'}]}
            cache.run(first,action,lambda *args:source)
            reused=cache.run(root/'attempt-0002',action,None)
            reused['agent_r_diagnostic']={'status':'completed','metrics':{'gain':.03}}
            reused['metrics'][0].update(agent_diagnostic_status='completed',agent_diagnostic_gain=.03)
            verify_reused_result(reused,root,{'sha256':'data'})
            altered=json.loads(json.dumps(reused));altered['metrics'][0]['agent_diagnostic_gain']=.5
            with self.assertRaises(ValueError):verify_reused_result(altered,root,{'sha256':'data'})
            altered=json.loads(json.dumps(reused));altered['metrics'][0]['n']='11'
            with self.assertRaises(ValueError):verify_reused_result(altered,root,{'sha256':'data'})
            altered=json.loads(json.dumps(reused));altered['estimates'][0]['coefficient']='.2'
            with self.assertRaises(ValueError):verify_reused_result(altered,root,{'sha256':'data'})
