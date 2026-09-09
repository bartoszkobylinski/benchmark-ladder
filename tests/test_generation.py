from benchmark_ladder.scoring import ExactMatchPolicy, exact_match, normalize_output


def test_exact_match_is_exact_by_default() -> None:
    policy = ExactMatchPolicy()
    assert exact_match(b"Answer", b"Answer", policy)
    assert not exact_match(b" Answer ", b"Answer", policy)


def test_normalization_is_explicit_and_idempotent() -> None:
    policy = ExactMatchPolicy(strip_outer_ascii_whitespace=True, ascii_case_insensitive=True)
    value = b"  HeLLo\n"
    once = normalize_output(value, policy)
    twice = normalize_output(once, policy)
    assert once == b"hello"
    assert twice == once
    assert exact_match(value, b"hello", policy)
