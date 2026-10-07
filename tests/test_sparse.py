import pytest

from steam_game_analyst.retrieval import sparse


def test_tokenize_is_casefolded_and_keeps_version_numbers():
    assert sparse.tokenize("Patch 2.0 fixed DLSS, ＡＭＤ too") == [
        "patch",
        "2",
        "0",
        "fixed",
        "dlss",
        "amd",  # NFKC folds full-width letters
        "too",
    ]


def test_tokenize_splits_chinese_and_japanese_into_bigrams_but_not_korean():
    assert sparse.tokenize("性能问题") == ["性能", "能问", "问题"]
    assert sparse.tokenize("バグ") == ["バグ"]
    assert sparse.tokenize("好") == ["好"]
    assert sparse.tokenize("크래시 문제") == ["크래시", "문제"]
    assert sparse.tokenize("dlss性能") == ["dlss", "性能"]  # mixed run without a space


def test_term_index_is_stable_across_processes():
    # stored vectors depend on it: changing the hash silently breaks an existing index
    assert sparse._index("dlss") == 431949890


def _weight(vec, token):
    return dict(zip(vec.indices, vec.values, strict=True))[sparse._index(token)]


def test_document_weights_saturate_with_term_frequency():
    once = _weight(sparse.encode_document("crash " + "x " * 9), "crash")
    thrice = _weight(sparse.encode_document("crash crash crash " + "x " * 7), "crash")
    assert once < thrice < sparse.K1 + 1


def test_document_weights_fall_with_length():
    short = _weight(sparse.encode_document("crash"), "crash")
    long = _weight(sparse.encode_document("crash " + "x " * 300), "crash")
    assert long < short


def test_vectors_have_sorted_unique_indices_and_unit_query_weights():
    doc = sparse.encode_document("b a b c a")
    assert doc.indices == sorted(set(doc.indices)) and len(doc.indices) == 3
    q = sparse.encode_query("steam deck steam")
    assert len(q.indices) == 2 and q.values == [1.0, 1.0]


@pytest.mark.parametrize("text", ["", "  ", "!!!"])
def test_empty_text_gives_an_empty_vector(text):
    assert sparse.encode_query(text).indices == []
    assert sparse.encode_document(text).indices == []
