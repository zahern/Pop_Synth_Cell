"""Focused tests for the structural relationship-in-household repair.

Run with:  python -X utf8 test_rlhp_repair.py
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mip_fit as mf


def _targets(shares):
    total = 10000
    return {("rlhp", int(k)): float(v) * total for k, v in shares.items()}


def _frame(rows):
    """rows: list of (household, role, age) or (household, role, age, stup)."""
    recs = []
    for i, row in enumerate(rows):
        hh, role, age = row[:3]
        rec = {"abshid": hh, "rlhp": role, "agep": age,
               "abspid": i, "stup": row[3] if len(row) > 3 else 1}
        recs.append(rec)
    return pd.DataFrame(recs)


def _households(df):
    return df.groupby("abshid").rlhp.apply(lambda s: sorted(s.tolist())).to_dict()


def test_six_partners_become_one_couple():
    df = _frame([("h", 1, a) for a in (40, 38, 36, 34, 32, 30)])
    tgt = _targets({1: 0.50, 5: 0.10, 6: 0.20, 7: 0.20})
    out, rep = mf.repair_rlhp_structure(df, tgt)
    roles = _households(out)["h"]
    assert roles.count(1) == 2, roles
    assert len(roles) == 6, roles
    assert rep["violations_after"]["more_than_two_partners"] == 0
    assert rep["violations_after"]["odd_partner_households"] == 0


def test_single_partner_becomes_lone_person():
    df = _frame([("h", 1, 30)])
    tgt = _targets({1: 0.30, 7: 0.70})
    out, rep = mf.repair_rlhp_structure(df, tgt)
    assert _households(out)["h"] == [7]
    assert rep["violations_after"]["one_person_wrong_role"] == 0


def test_lone_parent_without_child_is_reclassified():
    df = _frame([("h", 2, 30), ("h", 7, 32)])
    tgt = _targets({1: 0.30, 2: 0.30, 6: 0.20, 7: 0.20})
    out, rep = mf.repair_rlhp_structure(df, tgt)
    assert 2 not in _households(out)["h"]
    assert rep["violations_after"]["lone_parent_without_child"] == 0


def test_child_household_gets_a_parent():
    df = _frame([("h", 3, 8), ("h", 7, 35)])
    tgt = _targets({2: 0.20, 3: 0.40, 7: 0.40})
    out, rep = mf.repair_rlhp_structure(df, tgt)
    roles = _households(out)["h"]
    assert 2 in roles, roles
    assert 3 in roles, roles
    assert rep["violations_after"]["child_household_without_parent"] == 0


def test_under15_never_becomes_partner():
    df = _frame([("h", 1, 10), ("h", 1, 40), ("h", 1, 38)])
    tgt = _targets({1: 0.50, 3: 0.50})
    out, rep = mf.repair_rlhp_structure(df, tgt)
    roles = _households(out)["h"]
    assert 1 not in [r for r, a in zip(out.rlhp, out.agep) if a < 15]
    assert 3 in roles, roles
    assert rep["violations_after"]["under15_invalid_role"] == 0


def test_dependent_student_without_parent_is_reclassified():
    df = _frame([("h", 4, 20), ("h", 7, 22)])
    tgt = _targets({4: 0.50, 7: 0.50})
    out, rep = mf.repair_rlhp_structure(df, tgt)
    assert 4 not in _households(out)["h"]
    assert rep["violations_after"]["dependent_student_without_parent"] == 0


def test_student_promotion_respects_deficit():
    # One lone parent and one 18-year-old non-dependent child who is studying;
    # the target asks for one dependent student, so the 18-year-old is promoted.
    df = _frame([("h", 2, 45), ("h", 5, 18, 2)])
    tgt = _targets({2: 0.50, 4: 0.50})
    out, rep = mf.repair_rlhp_structure(df, tgt)
    assert _households(out)["h"] == [2, 4]
    assert rep["operations"]["students_to_dependent"] == 1


def test_student_promotion_is_capped():
    df = _frame([("h", 2, 45), ("h", 5, 18, 2), ("h", 5, 19, 2)])
    tgt = _targets({2: 0.50, 4: 0.25, 7: 0.25})
    out, rep = mf.repair_rlhp_structure(df, tgt)
    assert rep["operations"]["students_to_dependent"] == 1


def test_children_are_preserved():
    df = _frame([("h", 3, 6), ("h", 1, 38), ("h", 1, 36),
                 ("h", 3, 9), ("h", 4, 17, 2)])
    tgt = _targets({1: 0.40, 3: 0.30, 4: 0.30})
    out, _ = mf.repair_rlhp_structure(df, tgt)
    children = out[out.agep < 15].rlhp.tolist()
    assert children.count(3) == 2, children


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(tests)} tests passed")
