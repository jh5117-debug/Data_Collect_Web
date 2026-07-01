from __future__ import annotations


def support_query_disjoint(support_ids: set[str], query_ids: set[str]) -> bool:
    return not (support_ids & query_ids)


def fpr_safety_gate(base_fpr: float, adapted_fpr: float, *, max_abs_increase: float = 0.02, max_fpr: float = 0.03) -> bool:
    return adapted_fpr <= max_fpr and (adapted_fpr - base_fpr) <= max_abs_increase
