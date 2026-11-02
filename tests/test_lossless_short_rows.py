import json
from pathlib import Path
import tempfile
import unittest
from four_role_discovery import RoleHistory

class LosslessShortRowTests(unittest.TestCase):
    def test_repeated_short_tsv_rows_are_preserved_and_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            histories=RoleHistory(Path(directory))
            rows=['result-0001\ttypicality\t0.0194\t0.192\tX\t+' for _ in range(150)]
            prompt='\n'.join(rows)
            histories.add('hypothesis','decision',{'x':'fixture'},prompt)
            rendered=histories.prompt('hypothesis','CURRENT',shared_memory=False)
            dictionary,body=rendered.split('\nEND TEXT DICTIONARY\n',1)
            definitions={line.split('\t',1)[0]:line.split('\t',1)[1] for line in dictionary.split('\n')[1:]}
            decoded='\n'.join(definitions[line[6:-1]] if line.startswith('[SAME ') and line.endswith(']') else line for line in body.split('\n'))
            self.assertIn(prompt,decoded)
            self.assertIn('CURRENT',decoded)
            self.assertLess(len(rendered),len(decoded))
            self.assertEqual((histories.root/'hypothesis.jsonl').read_text().count('result-0001'),150)

    def test_shared_and_plain_roles_use_identical_lossless_encoding(self):
        import four_role_discovery as module
        with tempfile.TemporaryDirectory() as directory:
            histories=RoleHistory(Path(directory))
            histories.add('hypothesis','hypothesis',{'x':'fixture'},('same exact scientific row\t0.0194\t0.192\tX\n'*100))
            before=module.COMPACT_SHARED_HISTORY
            try:
                module.COMPACT_SHARED_HISTORY=False
                self.assertEqual(histories.prompt('hypothesis','CURRENT',shared_memory=False),histories.prompt('hypothesis','CURRENT',shared_memory=True))
            finally:module.COMPACT_SHARED_HISTORY=before
