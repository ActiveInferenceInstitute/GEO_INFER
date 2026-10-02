"""
Unit tests for TestOrchestrator using standard and property-based testing.
"""

from hypothesis import given, strategies as st

# Alias imports to avoid pytest collection warnings
from geo_infer_test.core.test_orchestrator import (
    TestOrchestrator as _TestOrchestrator,
    TestSuiteManager as _TestSuiteManager,
    TestSuiteDefinition as _TestSuiteDefinition,
)


class TestTestSuiteManager:
    """Standard tests for TestSuiteManager."""

    def test_default_suites(self):
        manager = _TestSuiteManager()
        suites = manager.list_suites()
        names = [s["name"] for s in suites]
        assert "unit" in names
        assert "integration" in names
        assert "performance" in names

    def test_register_and_combine_suites(self):
        manager = _TestSuiteManager()

        # Register custom suite
        custom = _TestSuiteDefinition(
            name="custom", description="desc", modules=["A", "B"]
        )
        manager.register_suite(custom)
        assert manager.get_suite("custom") is not None

        # Combine
        combined = manager.combine_suites(["unit", "custom"])
        assert "A" in combined.modules
        assert "B" in combined.modules
        assert len(combined.test_patterns) > 0


class TestTestOrchestrator:
    """Standard tests for TestOrchestrator."""

    def test_resolve_execution_order_simple(self):
        """Test topological sort with simple dependencies."""
        deps = {
            "A": ["B"],
            "B": ["C"],
            "C": [],
        }
        orchestrator = _TestOrchestrator(dependencies=deps)
        order = orchestrator.resolve_execution_order(["A", "C", "B"])

        # C must come before B, B must come before A
        assert order.index("C") < order.index("B")
        assert order.index("B") < order.index("A")

    def test_resolve_execution_order_diamond(self):
        """Test diamond dependency structure."""
        deps = {
            "A": ["B", "C"],
            "B": ["D"],
            "C": ["D"],
            "D": [],
        }
        orchestrator = _TestOrchestrator(dependencies=deps)
        order = orchestrator.resolve_execution_order(["A", "B", "C", "D"])

        assert order.index("D") < order.index("B")
        assert order.index("D") < order.index("C")
        assert order.index("B") < order.index("A")
        assert order.index("C") < order.index("A")


# ---------------------------------------------------------------------------
# Property-Based Tests (Hypothesis)
# ---------------------------------------------------------------------------


@st.composite
def dag_strategy(draw):
    """
    Strategy to generate random Directed Acyclic Graphs (DAGs).
    Returns a dict mapping nodes to list of dependencies.
    """
    nodes = draw(
        st.lists(
            st.text(
                min_size=1,
                max_size=5,
                alphabet=st.characters(whitelist_categories=("L",)),
            ),
            min_size=2,
            max_size=20,
            unique=True,
        )
    )
    deps = {n: [] for n in nodes}

    # Ensure acyclic by only allowing dependencies on nodes with higher index
    # (or lower, consistency just matters)
    for i, node in enumerate(nodes):
        # Can depend on any node that comes AFTER it in the list (reverse topo construction)
        possible_deps = nodes[i + 1 :]
        if possible_deps:
            chosen_deps = draw(
                st.lists(st.sampled_from(possible_deps), max_size=len(possible_deps))
            )
            deps[node] = list(set(chosen_deps))

    return deps


class TestHypothesisOrchestrator:
    """Property-based tests for dependency resolution."""

    @given(dag_strategy())
    def test_topological_sort_correctness(self, deps):
        """
        Verify topological sort always respects dependency constraints
        for any randomly generated DAG.
        """
        orchestrator = _TestOrchestrator(dependencies=deps)
        all_modules = list(deps.keys())

        # Sort
        order = orchestrator.resolve_execution_order(all_modules)

        # Verify: for every module M, all its dependencies D must appear BEFORE M in order
        index_map = {m: i for i, m in enumerate(order)}

        for module in order:
            dependencies = deps.get(module, [])
            for dep in dependencies:
                if dep in index_map:  # only check if dep is in the list
                    assert index_map[dep] < index_map[module], (
                        f"Dependency violation: {dep} came after {module}"
                    )

    @given(st.lists(st.text(min_size=1), min_size=1, max_size=10))
    def test_combine_suites_properties(self, suite_names):
        """Verify combined suites always contain the union of properties."""
        manager = _TestSuiteManager()
        # Register dummy suites first
        for name in suite_names:
            manager.register_suite(
                _TestSuiteDefinition(
                    name=name, description="desc", modules=[name], timeout_seconds=10
                )
            )

        combined = manager.combine_suites(suite_names)

        # Check properties
        assert len(combined.modules) == len(
            set(suite_names)
        )  # Should equal unique names count
        for name in set(suite_names):
            assert name in combined.modules


def test_dependency_cycle_fails_before_execution():
    import pytest

    with pytest.raises(ValueError, match="Cyclic test dependencies"):
        _TestOrchestrator(
            dependencies={"A": ["B"], "B": ["A"]}
        ).resolve_execution_order(["A", "B"])


def test_empty_and_custom_plan_require_actual_execution():
    import pytest

    orchestrator = _TestOrchestrator()
    with pytest.raises(ValueError, match="empty test plan"):
        orchestrator.execute_plan({"suite": "unit", "execution_order": []})
    with pytest.raises(ValueError, match="explicit runner_fn"):
        orchestrator.execute_plan({"suite": "custom", "execution_order": ["ACT"]})
    report = orchestrator.execute_plan(
        {"suite": "custom", "execution_order": ["ACT"]},
        runner_fn=lambda _module: {"status": "skipped"},
    )
    assert not report["success"] and report["failed"] == 1
