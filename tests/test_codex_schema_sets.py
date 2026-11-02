import unittest
from jsonschema import Draft202012Validator
from codex_role_backend import normalize_schema_sets
from two_role_discovery import decision_schema

class SchemaSetTests(unittest.TestCase):
    def test_duplicate_terms_removed_without_adding_or_changing_terms(self):
        schema={'type':'object','properties':{'terms':{'type':'array','items':{'enum':['typicality','previous_tags']},'uniqueItems':True}}}
        requested={'terms':['typicality','previous_tags','typicality']}
        value,edits=normalize_schema_sets(requested,schema)
        self.assertEqual(value['terms'],['typicality','previous_tags'])
        self.assertEqual(requested['terms'],['typicality','previous_tags','typicality'])
        self.assertTrue(edits)
        self.assertTrue(Draft202012Validator(schema).is_valid(value))

    def test_current_result_reordered_without_inventing_a_citation(self):
        schema=decision_schema('result-0002',{'result-0001','result-0002'})
        value={'action':'revise','basis_keywords':'fixture','next_inquiry':'fixture','evidence_ids':['result-0001','result-0002']}
        fixed,edits=normalize_schema_sets(value,schema)
        self.assertEqual(fixed['evidence_ids'],['result-0002','result-0001'])
        self.assertTrue(Draft202012Validator(schema).is_valid(fixed))
        value['evidence_ids']=['result-0001']
        fixed,_=normalize_schema_sets(value,schema)
        self.assertFalse(Draft202012Validator(schema).is_valid(fixed))
        self.assertEqual(fixed['evidence_ids'],['result-0001'])

    def test_nonset_tuple_order_and_unknown_scalar_are_not_repaired(self):
        schema={'type':'array','prefixItems':[{'const':'current'},{'const':'prior'}],'minItems':2,'maxItems':2}
        value=['prior','current']
        self.assertEqual(normalize_schema_sets(value,schema),(value,[]))
        self.assertFalse(Draft202012Validator(schema).is_valid(value))
