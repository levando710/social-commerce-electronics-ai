"""Observable two-way inference feedback; weights fixed before evaluation."""
from __future__ import annotations


from phone_assistant.retrieval.search.bm25 import tokenize
from phone_assistant.conversation.conversation_state import catalog_facts, constraint_verdict, next_plan
from phone_assistant.retrieval.search.hybrid_search import evidence_sentences
from phone_assistant.retrieval.serving.retrieval_service import retrieve

MODES = ("decoupled", "rag_to_rec", "rec_to_rag", "coupled")


class CoGuidingEngine:
    def __init__(self, documents, index, recommender, *, dense=None, method="legacy", joint_query=None,
                 planner=None, fact_extractor=None, verdict=None, retrieval=None):
        self.documents, self.index, self.recommender = documents, index, recommender
        self.dense, self.method = dense, method
        self.joint_query = joint_query
        self.retrieve = retrieval or retrieve
        self.by_id = {doc["doc_id"]: doc for doc in documents}
        self.planner = planner or next_plan
        self.verdict = verdict or constraint_verdict
        self.facts = {doc_id: (fact_extractor or catalog_facts)(doc) for doc_id, doc in self.by_id.items()}

    def run(self, query, *, user_index=None, previous_plan=None, mode="coupled", top_k=5,
            preference_feed=None, seen_item_ids=None):
        if mode not in MODES or not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 100:
            raise ValueError("Invalid coupling mode or top_k")
        # Default experiment order comes from the trained model; local browser profiles
        # can supply their separate, explicitly untrained preference baseline.
        original_feed = preference_feed if preference_feed is not None else self.recommender.recommend(user_index, top_k=100)
        plan = self.planner(query, previous_plan)
        retrieval_query = plan["query_rewrite"]
        base = self.retrieve(self.index, self.documents, retrieval_query, method=self.method,
                        dense=self.dense, top_k=80, use_llm=False)
        lexical_rank = {row["doc_id"]: rank for rank, row in enumerate(base, 1)}
        preference_rank = {row["doc_id"]: rank for rank, row in enumerate(original_feed["results"], 1)}
        candidate_ids = set(lexical_rank)
        focus_ids = set(plan.get('focus_doc_ids', [])) & self.by_id.keys()
        candidate_ids.update(focus_ids)
        joint_rank = {}
        if self.joint_query is not None:
            joint_rank = {doc_id: rank for rank,(doc_id,_) in enumerate(self.joint_query.rank(retrieval_query),1)}
            candidate_ids.update(joint_rank)
        rec_to_rag = mode in ("rec_to_rag", "coupled")
        rag_to_rec = mode in ("rag_to_rec", "coupled")
        if rec_to_rag:
            candidate_ids.update(list(preference_rank)[:40])
        if focus_ids:
            candidate_ids = focus_ids
        verdicts = {doc_id: self.verdict(self.facts[doc_id], plan["constraints"]) for doc_id in candidate_ids}
        ranked = []
        for doc_id in candidate_ids:
            if not verdicts[doc_id]["eligible"]:
                continue
            query_score = 1 / (60 + lexical_rank[doc_id]) if doc_id in lexical_rank else 0.0
            prior_score = .2 / (60 + preference_rank[doc_id]) if rec_to_rag and doc_id in preference_rank else 0.0
            shared_score = .5/(60+joint_rank[doc_id]) if doc_id in joint_rank else 0.
            ranked.append((doc_id, query_score + prior_score + shared_score + (1.0 if doc_id in focus_ids else 0.0)))
        ranked.sort(key=lambda row: (-row[1], row[0]))
        rows = []
        for doc_id, value in ranked[:top_k]:
            doc = self.by_id[doc_id]
            rows.append({"doc_id": doc_id, "score": value, "title": doc.get("title"),
                "brand": doc.get("brand"), "average_rating": doc.get("average_rating"),
                "review_count": doc.get("review_count"), "product_images": doc.get("product_image_paths", [])[:2],
                "evidence": evidence_sentences(doc, tokenize(retrieval_query)),
                "catalog_facts": self.facts[doc_id], "constraint_check": verdicts[doc_id],
                "query_plan": plan,
                "feedback": {"query_rank": lexical_rank.get(doc_id),
                             "joint_query_rank":joint_rank.get(doc_id),
                             "preference_rank": preference_rank.get(doc_id) if rec_to_rag else None}})
        feed = list(original_feed["results"])
        plan['last_result_ids'] = [row['doc_id'] for row in rows]
        if rag_to_rec:
            # RAG rewards evidence-bearing items and excludes unverified hard constraints.
            evidence_rank = {row["doc_id"]: rank for rank, row in enumerate(rows, 1)}
            feed = []
            base_feed = list(original_feed["results"])
            base_ids = {row["doc_id"] for row in base_feed}
            base_feed += [{"doc_id": row["doc_id"], "title": row["title"], "brand": row["brand"],
                           "score": 0.0, "product_images": row["product_images"]}
                          for row in rows if row["doc_id"] not in base_ids]
            seen = self.recommender.data.train_seen[user_index] if user_index is not None else set()
            for rank, item in enumerate(base_feed, 1):
                doc_id = item["doc_id"]
                index = self.recommender.data.maps["items"][doc_id]
                if index in seen or doc_id in (seen_item_ids or ()) or not self.verdict(self.facts[doc_id], plan["constraints"])["eligible"]:
                    continue
                reward = .03 / evidence_rank[doc_id] if doc_id in evidence_rank else 0.0
                feed.append({**item, "base_score": item["score"], "intent_reward": reward,
                             "score": 1 / (60 + rank) + reward})
            feed.sort(key=lambda item: (-item["score"], item["doc_id"]))
        return {"query": query, "query_plan": plan, "results": rows,
                "recommendations": feed[:top_k], "coupling_mode": mode,
                "feedback": {"rec_to_rag": rec_to_rag, "rag_to_rec": rag_to_rec,
                    "personalized": original_feed["personalized"], "base_candidates": len(base),
                    "joint_ranking":self.joint_query is not None,
                    "eligible_candidates": len(ranked),
                    "rejected_unknown": sum(bool(v["unknown"]) for v in verdicts.values()),
                    "rejected_violations": sum(bool(v["violations"]) for v in verdicts.values())},
                "abstention": not rows}
