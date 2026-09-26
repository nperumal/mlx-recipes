"""Correctness tests for <recipe name>.

Four kinds, none needing an external library — see CONTRIBUTING.md §4:

    test_synthetic_recovery   generate from known parameters, assert recovery
    test_paths_agree          closed form and gradient descent reach the same answer
    test_gradient_check       finite differences vs mx.grad
    test_known_answer         a tiny case worked out by hand

See recipes/foundations/linear-regression/tests/test_correctness.py for worked
versions of all four.

Optionally one more, which must skip cleanly without dev extras:

    @pytest.mark.reference
    def test_matches_reference(): ...
"""
