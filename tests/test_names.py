"""케이스 **이름**에 대한 검사.

값이 아니라 표 자체를 본다. 골든은 이름을 열쇠로 쓰는 딕셔너리에 담기는데, 표는
리스트다. 그 둘 사이에서 조용히 사라지는 것이 있으면 여기서 걸린다.
"""

import collections

import cases as cases_mod


def test_case_names_are_unique():
    """이름이 겹치면 **먼저 온 케이스가 사라진다.**

    `golden.dump` 는 `data[name] = ...` 로 담고 `export_json` 도 딕셔너리로 담는다.
    이름이 같은 케이스가 둘이면 나중 것이 먼저 것을 덮어쓰고, 먼저 것의 기대값은
    어디에도 남지 않는다. 표에는 케이스가 있는데 아무도 그걸 안 묻는 상태가 되고,
    개수만 보면 눈치챌 수 없다 — dump 는 리스트 길이(겹친 것 포함)를 세고 대조는
    딕셔너리 길이(겹친 것 제외)를 센다.
    """
    names = [name for name, _ in cases_mod.golden_cases(cases_mod.golden_inputs())]
    dup = [n for n, c in collections.Counter(names).items() if c > 1]
    assert not dup, "케이스 이름이 겹친다 — 먼저 온 것이 덮여 사라진다:\n  " + "\n  ".join(dup)
