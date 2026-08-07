from __future__ import annotations

from memoria.claims import Claim, extract_claims


def test_extract_claims_keeps_only_explicit_preference_statements() -> None:
    assert extract_claims("I no longer prefer coffee in the morning.") == (
        Claim(predicate="preference", value_key="coffee", polarity=-1),
    )
    assert extract_claims("Perhaps coffee is popular.") == ()


def test_extract_claims_normalizes_equivalent_preference_values() -> None:
    assert extract_claims("I prefer Coffee!") == (
        Claim(predicate="preference", value_key="coffee", polarity=1),
    )


def test_extract_claims_supports_explicit_exclusive_profile_facts() -> None:
    assert extract_claims("I live in Berlin.") == (
        Claim(predicate="location", value_key="berlin", polarity=1, exclusive=True),
    )
    assert extract_claims("I work at Example Labs.") == (
        Claim(predicate="employer", value_key="example labs", polarity=1, exclusive=True),
    )
