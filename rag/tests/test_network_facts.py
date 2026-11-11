from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phone_assistant.conversation.conversation_state import catalog_facts, constraint_verdict


def doc(title, carrier=None):
    return {"title": title, "details": {"Wireless Carrier": carrier} if carrier else {}}


class NetworkFactTests(unittest.TestCase):
    def test_conflicting_title_and_carrier_stay_unverified(self):
        facts = catalog_facts(doc("Apple iPhone 7 AT&T 128 GB Locked to AT&T", "Unlocked for All Carriers"))
        self.assertIsNone(facts["network"])
        verdict = constraint_verdict(facts, {"network": "unlocked"})
        self.assertFalse(verdict["eligible"])
        self.assertIn("network", verdict["unknown"])

    def test_consistent_evidence_is_unchanged(self):
        self.assertEqual(catalog_facts(doc("Phone 128GB Unlocked"))["network"], "unlocked")
        self.assertEqual(catalog_facts(doc("Phone", "Unlocked for All Carriers"))["network"], "unlocked")
        self.assertEqual(catalog_facts(doc("Phone Verizon Locked (Renewed)"))["network"], "locked")
        self.assertEqual(catalog_facts(doc("Phone Not Unlocked"))["network"], "locked")
        self.assertIsNone(catalog_facts(doc("Phone", "AT&T"))["network"])


if __name__ == "__main__":
    unittest.main()
