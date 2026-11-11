import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phone_assistant.conversation.conversation_state import SessionStore, catalog_facts, constraint_verdict, next_plan
from phone_assistant.evaluation.metrics.answer_evaluation import parse_judgment, summarize_claims
from unittest.mock import patch


class ConversationTests(unittest.TestCase):
    def test_reference_to_previous_result_preserves_item_identity(self):
        first=next_plan('Samsung')
        first['last_result_ids']=['a','b','c']
        self.assertEqual(next_plan('first one battery',first)['focus_doc_ids'],['a'])
        self.assertEqual(next_plan('compare them',first)['focus_doc_ids'],['a','b'])
        self.assertEqual(next_plan('new search nokia',first)['focus_doc_ids'],[])
    def test_followup_replaces_storage_and_keeps_brand_and_model(self):
        first = next_plan('iPhone 13 128GB unlocked')
        second = next_plan('con 256GB thi sao', first)
        self.assertEqual(second['constraints']['brands'], ['apple'])
        self.assertEqual(second['constraints']['model_terms'], 'iphone 13')
        self.assertEqual(second['constraints']['storage_gb'], 256)
        self.assertEqual(second['constraints']['network'], 'unlocked')
        self.assertNotIn('locked', second['tags'])

    def test_brand_switch_clears_old_model_and_reset_clears_constraints(self):
        first = next_plan('iphone 13 128GB')
        second = next_plan('Samsung', first)
        self.assertEqual(second['constraints']['brands'], ['samsung'])
        self.assertNotIn('model_terms', second['constraints'])
        reset = next_plan('new search nokia', second)
        self.assertNotIn('storage_gb', reset['constraints'])

    def test_boundary_matching_does_not_confuse_bluetooth_and_blu(self):
        plan = next_plan('bluetooth unlocked')
        self.assertNotIn('blu', plan['constraints'].get('brands', []))
        self.assertEqual(plan['constraints']['network'], 'unlocked')

    def test_ram_and_storage_are_separate(self):
        plan = next_plan('Samsung 8GB RAM 128GB')
        self.assertEqual(plan['constraints']['ram_gb'], 8)
        self.assertEqual(plan['constraints']['storage_gb'], 128)

    def test_review_cannot_prove_hard_storage_or_network(self):
        facts = catalog_facts({'title': 'Samsung phone', 'details': {},
                              'text': 'review says unlocked and 128GB'})
        verdict = constraint_verdict(facts, {'storage_gb': 128, 'network': 'unlocked'})
        self.assertFalse(verdict['eligible'])
        self.assertEqual(set(verdict['unknown']), {'storage_gb', 'network'})

    def test_missing_currency_price_abstains_instead_of_converting_amazon_price(self):
        facts = catalog_facts({'title': 'Samsung 128GB unlocked', 'details': {}})
        plan = next_plan('Samsung duoi 5 trieu')
        self.assertEqual(plan['constraints']['max_price'], 5000000)
        self.assertFalse(constraint_verdict(facts, plan['constraints'])['eligible'])

    def test_session_isolation_profile_binding_and_failed_transaction_cleanup(self):
        store = SessionStore()
        with store.transaction(user_index=0) as (first, session):
            session.plan = next_plan('iphone')
        with store.transaction(user_index=1) as (second, session):
            self.assertEqual(session.plan, {})
        self.assertNotEqual(first, second)
        with self.assertRaises(ValueError):
            with store.transaction(first, 1):
                pass
        before = len(store.sessions)
        with self.assertRaises(RuntimeError):
            with store.transaction():
                raise RuntimeError('failed request')
        self.assertEqual(len(store.sessions), before)


class AnswerJudgeTests(unittest.TestCase):
    def test_fake_supported_quote_is_unknown_and_not_a_perfect_score(self):
        result = parse_judgment('{"claims":[{"text":"128GB", "supported":true, "doc_id":"a", "quote":"128GB"}], "reverse_questions":["storage?"]}', {'a':'64GB'})
        metrics = summarize_claims(result['claims'])
        self.assertIsNone(metrics['faithfulness'])
        self.assertEqual(metrics['unknown'], 1)

    def test_supported_and_unsupported_claim_denominators(self):
        result = summarize_claims([{'supported': True}, {'supported': False}, {'supported': None}])
        self.assertEqual(result['faithfulness'], .5)
        self.assertEqual(result['context_unsupported_rate'], .5)
        self.assertAlmostEqual(result['judge_coverage'], 2/3)

    def test_no_claims_is_not_automatic_faithfulness_one(self):
        self.assertIsNone(summarize_claims([])['faithfulness'])


class JointObjectiveTests(unittest.TestCase):
    def test_shared_retrieval_loss_updates_query_and_graph_parameters(self):
        import numpy as np
        import torch
        from types import SimpleNamespace
        from phone_assistant.recommendation.joint_ranking import JointGraphRanker
        data=SimpleNamespace(n_users=2,n_items=3,item_ids=['a','b','c'],pairs={'train':np.array([[0,0],[1,1]])})
        features={'text':np.eye(3,dtype=np.float32),'image':np.eye(3,dtype=np.float32),
                  'image_mask':np.array([True,True,True]),'specs':{'a':['brand=a'],'b':['brand=b'],'c':['brand=a']}}
        torch.manual_seed(42)
        model=JointGraphRanker(data,features,dim=4)
        losses=model.loss(torch.tensor([0,1]),torch.tensor([0,1]),torch.tensor([2,2]))
        self.assertTrue(torch.isfinite(losses['bidirectional_js']))
        self.assertGreaterEqual(float(losses['bidirectional_js'].detach()),-1e-6)
        losses['total'].backward()
        self.assertGreater(float(model.query_projection.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.core.user_embedding.weight.grad.abs().sum()),0)


class GeminiClientTests(unittest.TestCase):
    def test_network_error_does_not_expose_key(self):
        import requests
        from phone_assistant.llm.gemini_client import call_gemini
        with patch('phone_assistant.llm.gemini_client.get_secret',return_value='TEST_SECRET'),\
             patch('phone_assistant.llm.gemini_client.gemini_model_name',return_value='gemini-2.5-flash'),\
             patch('phone_assistant.llm.gemini_client.requests.post',side_effect=requests.RequestException('url?key=TEST_SECRET')):
            result=call_gemini([],retries=0)
        self.assertNotIn('TEST_SECRET',result['error'])

    def test_judge_json_and_thinking_config_and_api_error_are_handled(self):
        from types import SimpleNamespace
        from phone_assistant.llm.gemini_client import call_gemini
        response=SimpleNamespace(status_code=503,text='unavailable',headers={})
        with patch('phone_assistant.llm.gemini_client.get_secret',return_value='TEST_SECRET'),\
             patch('phone_assistant.llm.gemini_client.gemini_model_name',return_value='gemini-2.5-flash'),\
             patch('phone_assistant.llm.gemini_client.requests.post',return_value=response) as request:
            result=call_gemini([],retries=0,thinking_budget=512,response_mime_type='application/json')
        self.assertIn('503',result['error'])
        config=request.call_args.kwargs['json']['generationConfig']
        self.assertEqual(config['thinkingConfig']['thinkingBudget'],512)
        self.assertEqual(config['responseMimeType'],'application/json')


if __name__ == '__main__':
    unittest.main()
