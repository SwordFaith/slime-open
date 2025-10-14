"""
Component Registry Unit Tests

Tests cover component registry functionality:
- Basic operations (register, get, remove, list)
- Thread safety under extreme contention (1000+ threads)
- Error handling and validation
- Destructor invocation
- Transaction support
- Edge cases (empty names, None values, etc.)

Test Strategy:
- Unit testing with minimal mocking
- Extreme concurrency testing (1000+ threads)
- Resource cleanup verification
- Error condition testing
"""

import pytest
import threading
import time
from slime.router.utils.component_registry import ComponentRegistry


# ==============================================================================
# Group A: Basic Operations
# ==============================================================================

@pytest.mark.unit
class TestBasicOperations:
    """Test basic registry operations."""

    def test_register_and_get(self):
        """
        Test: Register component and retrieve it.

        Expected: Retrieved component matches registered one
        """
        registry = ComponentRegistry()
        component = {"data": "test_component"}

        registry.register("test", component)
        retrieved = registry.get("test")

        assert retrieved is component, "Should return same instance"

    def test_register_multiple_components(self):
        """
        Test: Register multiple different components.

        Expected: All can be retrieved independently
        """
        registry = ComponentRegistry()

        comp1 = {"name": "component1"}
        comp2 = {"name": "component2"}
        comp3 = {"name": "component3"}

        registry.register("comp1", comp1)
        registry.register("comp2", comp2)
        registry.register("comp3", comp3)

        assert registry.get("comp1") is comp1
        assert registry.get("comp2") is comp2
        assert registry.get("comp3") is comp3

    def test_has_component(self):
        """
        Test: Check if component exists.

        Expected: Returns True for registered, False for missing
        """
        registry = ComponentRegistry()
        registry.register("exists", {"data": "test"})

        assert registry.has("exists") is True
        assert registry.has("missing") is False

    def test_remove_component(self):
        """
        Test: Remove registered component.

        Expected: Component no longer accessible after removal
        """
        registry = ComponentRegistry()
        registry.register("to_remove", {"data": "test"})

        # Remove component
        result = registry.remove("to_remove")
        assert result is True, "Remove should return True"

        # Should no longer exist
        assert registry.has("to_remove") is False

    def test_remove_nonexistent(self):
        """
        Test: Remove component that doesn't exist.

        Expected: Returns False
        """
        registry = ComponentRegistry()

        result = registry.remove("nonexistent")
        assert result is False, "Should return False for missing component"

    def test_list_components(self):
        """
        Test: List all registered components.

        Expected: Returns list of component names
        """
        registry = ComponentRegistry()
        registry.register("comp1", {})
        registry.register("comp2", {})
        registry.register("comp3", {})

        names = registry.list_components()
        assert len(names) == 3
        assert "comp1" in names
        assert "comp2" in names
        assert "comp3" in names

    def test_list_dict(self):
        """
        Test: Get dict of all components.

        Expected: Returns copy of components dict
        """
        registry = ComponentRegistry()
        comp1 = {"name": "first"}
        comp2 = {"name": "second"}

        registry.register("comp1", comp1)
        registry.register("comp2", comp2)

        components = registry.list()
        assert len(components) == 2
        assert components["comp1"] is comp1
        assert components["comp2"] is comp2

    def test_clear(self):
        """
        Test: Clear all components.

        Expected: Registry becomes empty
        """
        registry = ComponentRegistry()
        registry.register("comp1", {})
        registry.register("comp2", {})

        registry.clear()

        assert len(registry.list_components()) == 0


# ==============================================================================
# Group B: Error Handling and Validation
# ==============================================================================

@pytest.mark.unit
class TestErrorHandlingAndValidation:
    """Test error conditions and input validation."""

    def test_register_empty_name(self):
        """
        Test: Register with empty name.

        Expected: ValueError
        """
        registry = ComponentRegistry()

        with pytest.raises(ValueError, match="cannot be empty"):
            registry.register("", {"data": "test"})

    def test_register_whitespace_name(self):
        """
        Test: Register with whitespace-only name.

        Expected: ValueError
        """
        registry = ComponentRegistry()

        with pytest.raises(ValueError, match="cannot be empty"):
            registry.register("   ", {"data": "test"})

    def test_register_none_instance(self):
        """
        Test: Register None as component instance.

        Expected: ValueError
        """
        registry = ComponentRegistry()

        with pytest.raises(ValueError, match="cannot be None"):
            registry.register("test", None)

    def test_register_duplicate_name(self):
        """
        Test: Register same name twice.

        Expected: RuntimeError on second registration
        """
        registry = ComponentRegistry()
        registry.register("duplicate", {"data": "first"})

        with pytest.raises(RuntimeError, match="already registered"):
            registry.register("duplicate", {"data": "second"})

    def test_get_nonexistent_no_default(self):
        """
        Test: Get component that doesn't exist without default.

        Expected: RuntimeError with helpful message
        """
        registry = ComponentRegistry()

        with pytest.raises(RuntimeError, match="not found"):
            registry.get("nonexistent")

    def test_get_nonexistent_with_default(self):
        """
        Test: Get component that doesn't exist with default.

        Expected: Returns default value
        """
        registry = ComponentRegistry()
        default = {"default": "value"}

        result = registry.get("nonexistent", default)
        assert result is default

    def test_get_empty_name(self):
        """
        Test: Get with empty name.

        Expected: ValueError
        """
        registry = ComponentRegistry()

        with pytest.raises(ValueError, match="cannot be empty"):
            registry.get("")


# ==============================================================================
# Group C: Thread Safety - Basic Concurrency
# ==============================================================================

@pytest.mark.unit
class TestThreadSafetyBasic:
    """Test thread safety with moderate concurrency."""

    def test_concurrent_register_different_names(self):
        """
        Test: Multiple threads register different components.

        Expected: All registrations succeed
        """
        registry = ComponentRegistry()
        results = []
        errors = []

        def worker(worker_id):
            try:
                component = {"id": worker_id, "data": f"data_{worker_id}"}
                registry.register(f"comp_{worker_id}", component)
                results.append(worker_id)
            except Exception as e:
                errors.append((worker_id, str(e)))

        # 100 concurrent threads
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should succeed
        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 100
        assert len(registry.list_components()) == 100

    def test_concurrent_get_operations(self):
        """
        Test: Multiple threads read same component.

        Expected: All get same instance
        """
        registry = ComponentRegistry()
        component = {"shared": "data"}
        registry.register("shared", component)

        results = []

        def worker(worker_id):
            retrieved = registry.get("shared")
            results.append((worker_id, retrieved is component))

        # 100 concurrent readers
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should get same instance
        assert all(is_same for _, is_same in results)

    def test_concurrent_mixed_operations(self):
        """
        Test: Mix of register, get, has, remove operations.

        Expected: No crashes, operations are atomic
        """
        registry = ComponentRegistry()

        # Pre-register some components
        for i in range(10):
            registry.register(f"initial_{i}", {"id": i})

        results = {"register": 0, "get": 0, "has": 0, "remove": 0, "errors": []}
        lock = threading.Lock()

        def worker(worker_id):
            try:
                # Register
                registry.register(f"worker_{worker_id}", {"id": worker_id})
                with lock:
                    results["register"] += 1

                # Get
                if registry.has(f"initial_{worker_id % 10}"):
                    registry.get(f"initial_{worker_id % 10}")
                    with lock:
                        results["get"] += 1

                # Has check
                registry.has(f"worker_{worker_id}")
                with lock:
                    results["has"] += 1

            except Exception as e:
                with lock:
                    results["errors"].append(str(e))

        # 50 concurrent workers
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should occur
        assert len(results["errors"]) == 0, f"Errors: {results['errors']}"


# ==============================================================================
# Group D: Extreme Contention (1000+ threads)
# ==============================================================================

@pytest.mark.unit
class TestExtremeContention:
    """Test thread safety under extreme concurrency."""

    def test_extreme_concurrent_register(self):
        """
        Test: 1000 threads register components simultaneously.

        Extreme Stress: Maximum contention on registry lock
        Expected: All registrations succeed, no corruption
        """
        registry = ComponentRegistry()
        results = []
        errors = []

        def worker(worker_id):
            try:
                registry.register(f"thread_{worker_id}", {"id": worker_id})
                results.append(worker_id)
            except Exception as e:
                errors.append((worker_id, str(e)))

        # 1000 concurrent threads
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(1000)]

        start_time = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        duration = time.time() - start_time

        # Verify results
        assert len(errors) == 0, f"Should have no errors, got {len(errors)}: {errors[:5]}"
        assert len(results) == 1000, f"All threads should succeed, got {len(results)}"
        assert len(registry.list_components()) == 1000, f"All components should be registered"

        print(f"1000 concurrent registrations completed in {duration:.3f}s")

    def test_extreme_concurrent_reads(self):
        """
        Test: 2000 threads read same component simultaneously.

        Extreme Stress: Read lock contention
        Expected: All reads succeed, return same instance
        """
        registry = ComponentRegistry()
        shared_component = {"shared": "data", "id": 12345}
        registry.register("shared", shared_component)

        results = []
        errors = []

        def worker(worker_id):
            try:
                component = registry.get("shared")
                results.append(component is shared_component)
            except Exception as e:
                errors.append((worker_id, str(e)))

        # 2000 concurrent readers
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(2000)]

        start_time = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        duration = time.time() - start_time

        # Verify results
        assert len(errors) == 0, f"No errors expected: {errors[:5]}"
        assert len(results) == 2000
        assert all(results), "All should get same instance"

        print(f"2000 concurrent reads completed in {duration:.3f}s")

    def test_extreme_mixed_operations(self):
        """
        Test: 1000 threads performing mixed operations.

        Extreme Stress: Mix of register, get, has, remove
        Expected: Atomic operations, no data corruption
        """
        registry = ComponentRegistry()

        # Pre-populate
        for i in range(100):
            registry.register(f"base_{i}", {"id": i})

        results = {"ops": 0, "errors": []}
        lock = threading.Lock()

        def worker(worker_id):
            try:
                # Mix of operations
                if worker_id % 4 == 0:
                    # Register
                    registry.register(f"new_{worker_id}", {"id": worker_id})
                elif worker_id % 4 == 1:
                    # Get
                    if registry.has(f"base_{worker_id % 100}"):
                        registry.get(f"base_{worker_id % 100}")
                elif worker_id % 4 == 2:
                    # Has check
                    registry.has(f"any_{worker_id}")
                else:
                    # List
                    registry.list_components()

                with lock:
                    results["ops"] += 1

            except Exception as e:
                with lock:
                    results["errors"].append(f"Worker {worker_id}: {str(e)}")

        # 1000 concurrent workers
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(1000)]

        start_time = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        duration = time.time() - start_time

        # Check results
        assert len(results["errors"]) == 0, f"Errors: {results['errors'][:10]}"
        assert results["ops"] == 1000

        print(f"1000 mixed operations completed in {duration:.3f}s")

    def test_extreme_register_remove_cycle(self):
        """
        Test: Rapid register/remove cycles with 500 threads.

        Extreme Stress: Contention on same component names
        Expected: No crashes, operations are atomic
        """
        registry = ComponentRegistry()

        results = {"registers": 0, "removes": 0, "errors": []}
        lock = threading.Lock()

        def worker(worker_id):
            try:
                # Use only 10 component names to create contention
                comp_name = f"contested_{worker_id % 10}"

                # Try to register (may fail if already registered)
                try:
                    registry.register(comp_name, {"id": worker_id})
                    with lock:
                        results["registers"] += 1
                except RuntimeError:
                    # Already registered, try to remove
                    if registry.remove(comp_name):
                        with lock:
                            results["removes"] += 1

            except Exception as e:
                with lock:
                    results["errors"].append(str(e))

        # 500 threads competing for 10 component names
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(500)]

        start_time = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        duration = time.time() - start_time

        # Should have some registers and removes
        assert results["registers"] > 0, "Should have some successful registers"
        assert results["removes"] >= 0, "May have some successful removes"
        assert len(results["errors"]) == 0, f"No errors expected: {results['errors'][:5]}"

        print(f"500 threads, register/remove contention completed in {duration:.3f}s")


# ==============================================================================
# Group E: Destructor Functionality
# ==============================================================================

@pytest.mark.unit
class TestDestructorFunctionality:
    """Test destructor invocation."""

    def test_destructor_called_on_remove(self):
        """
        Test: Destructor called when component is removed.

        Expected: Destructor invoked with component instance
        """
        registry = ComponentRegistry()

        destroyed = []

        def destructor(component):
            destroyed.append(component["id"])

        component = {"id": 42}
        registry.register("test", component, on_destroy=destructor)

        # Remove component
        registry.remove("test")

        # Destructor should have been called
        assert 42 in destroyed

    def test_destructor_called_on_clear(self):
        """
        Test: All destructors called when clearing.

        Expected: All destructors invoked
        """
        registry = ComponentRegistry()

        destroyed = []

        def make_destructor(expected_id):
            return lambda comp: destroyed.append(expected_id)

        # Register with destructors
        registry.register("comp1", {"id": 1}, on_destroy=make_destructor(1))
        registry.register("comp2", {"id": 2}, on_destroy=make_destructor(2))
        registry.register("comp3", {"id": 3}, on_destroy=make_destructor(3))

        # Clear all
        registry.clear()

        # All destructors should be called
        assert set(destroyed) == {1, 2, 3}

    def test_destructor_exception_handling(self):
        """
        Test: Destructor that raises exception.

        Expected: Exception caught, doesn't break cleanup
        """
        registry = ComponentRegistry()

        def bad_destructor(comp):
            raise Exception("Destructor error")

        registry.register("test", {"id": 1}, on_destroy=bad_destructor)

        # Remove should not raise (error is logged)
        try:
            registry.remove("test")
        except Exception:
            pytest.fail("Destructor exception should be caught")

        # Component should be removed despite error
        assert not registry.has("test")


# ==============================================================================
# Group F: Transaction Support
# ==============================================================================

@pytest.mark.unit
class TestTransactionSupport:
    """Test transaction context manager."""

    def test_transaction_context(self):
        """
        Test: Transaction provides atomic access.

        Expected: Multiple operations under same lock
        """
        registry = ComponentRegistry()
        registry.register("comp1", {"value": 1})
        registry.register("comp2", {"value": 2})

        # Use transaction
        with registry.transaction() as reg:
            comp1 = reg.get("comp1")
            comp2 = reg.get("comp2")

            assert comp1["value"] == 1
            assert comp2["value"] == 2

    def test_transaction_error_handling(self):
        """
        Test: Error in transaction releases lock.

        Expected: Lock released even if exception occurs
        """
        registry = ComponentRegistry()

        try:
            with registry.transaction() as reg:
                raise Exception("Test error")
        except Exception:
            pass

        # Should still be able to use registry
        registry.register("after_error", {"data": "test"})
        assert registry.has("after_error")


# ==============================================================================
# Group G: Statistics and Monitoring
# ==============================================================================

@pytest.mark.unit
class TestStatisticsAndMonitoring:
    """Test statistics functionality."""

    def test_get_stats_basic(self):
        """
        Test: Get basic registry statistics.

        Expected: Accurate counts
        """
        registry = ComponentRegistry()
        registry.register("comp1", {})
        registry.register("comp2", {}, on_destroy=lambda x: None)

        stats = registry.get_stats()

        assert stats["total_components"] == 2
        assert stats["components_with_destructors"] == 1
        assert "comp1" in stats["component_names"]
        assert "comp2" in stats["component_names"]

    def test_get_stats_empty(self):
        """
        Test: Statistics on empty registry.

        Expected: All counts zero
        """
        registry = ComponentRegistry()

        stats = registry.get_stats()

        assert stats["total_components"] == 0
        assert stats["components_with_destructors"] == 0
        assert len(stats["component_names"]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
