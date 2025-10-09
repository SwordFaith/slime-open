"""
Thread safety tests for ComponentRegistry.

This module tests the thread safety of ComponentRegistry under various concurrent scenarios:
- Multiple threads registering components simultaneously
- Multiple threads accessing components simultaneously
- Mixed read/write operations from multiple threads
- Edge cases like component overwriting and removal
"""

import pytest
import threading
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock

from slime.router.component_registry import ComponentRegistry


class MockComponent:
    """Mock component for testing."""

    def __init__(self, name: str):
        self.name = name
        self.created_at = time.time()
        self.access_count = 0
        self.lock = threading.Lock()

    def access(self):
        """Simulate component access."""
        with self.lock:
            self.access_count += 1
            return f"Component {self.name} accessed {self.access_count} times"


class TestComponentRegistryThreadSafety:
    """Test thread safety of ComponentRegistry."""

    def test_concurrent_registration_same_component(self):
        """Test multiple threads trying to register the same component."""
        registry = ComponentRegistry()
        component_name = "test_component"

        def register_component(thread_id: int):
            """Register component from a thread."""
            component = MockComponent(f"{component_name}_thread_{thread_id}")
            try:
                registry.register(component_name, component)
                return (thread_id, True, component.name)
            except Exception as e:
                return (thread_id, False, str(e))

        # Launch multiple threads registering the same component
        num_threads = 10
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(register_component, i) for i in range(num_threads)]
            results = [future.result() for future in as_completed(futures)]

        # Analyze results - all registrations should succeed (overwrites are allowed)
        successful_regs = [r for r in results if r[1]]
        failed_regs = [r for r in results if not r[1]]

        # All registrations should succeed (current implementation allows overwrites)
        assert len(successful_regs) == num_threads, f"Expected {num_threads} successful registrations, got {len(successful_regs)}"
        assert len(failed_regs) == 0, f"Expected 0 failed registrations, got {len(failed_regs)}"

        # The component should be accessible
        component = registry.get(component_name)
        assert isinstance(component, MockComponent)

        # The final component should be from one of the threads (race condition is expected)
        component_names = [r[2] for r in successful_regs]
        assert component.name in component_names

    def test_concurrent_registration_different_components(self):
        """Test multiple threads registering different components."""
        registry = ComponentRegistry()
        num_threads = 20
        component_names = [f"component_{i}" for i in range(num_threads)]

        def register_component(name: str):
            """Register a component."""
            component = MockComponent(name)
            registry.register(name, component)
            return name

        # Register components concurrently
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(register_component, name) for name in component_names]
            results = [future.result() for future in as_completed(futures)]

        # All registrations should succeed
        assert len(results) == num_threads
        assert set(results) == set(component_names)

        # All components should be accessible
        for name in component_names:
            component = registry.get(name)
            assert isinstance(component, MockComponent)
            assert component.name == name

    def test_concurrent_access_read_only(self):
        """Test multiple threads reading the same component concurrently."""
        registry = ComponentRegistry()
        component_name = "shared_component"

        # Register a component
        component = MockComponent(component_name)
        registry.register(component_name, component)

        def access_component(thread_id: int, num_accesses: int):
            """Access component multiple times."""
            results = []
            for i in range(num_accesses):
                comp = registry.get(component_name)
                results.append(comp.access())
                # Small random delay to increase chance of race conditions
                time.sleep(random.uniform(0.001, 0.005))
            return (thread_id, results)

        # Multiple threads accessing the component
        num_threads = 10
        accesses_per_thread = 50
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(access_component, i, accesses_per_thread)
                for i in range(num_threads)
            ]
            results = [future.result() for future in as_completed(futures)]

        # Verify all accesses were successful
        total_accesses = sum(len(result[1]) for result in results)
        expected_total = num_threads * accesses_per_thread
        assert total_accesses == expected_total

        # Component should have correct access count
        final_component = registry.get(component_name)
        assert final_component.access_count == total_accesses

    def test_mixed_read_write_operations(self):
        """Test mixed read/write operations from multiple threads."""
        registry = ComponentRegistry()
        num_components = 5
        component_names = [f"mixed_component_{i}" for i in range(num_components)]

        # Pre-register some components
        for name in component_names[:3]:
            registry.register(name, MockComponent(name))

        def mixed_operations(thread_id: int, num_operations: int):
            """Perform mixed read/write operations."""
            results = []
            for i in range(num_operations):
                operation = random.choice(['register', 'get', 'has', 'list'])

                if operation == 'register':
                    # Try to register a new component
                    name = f"thread_{thread_id}_component_{i}"
                    try:
                        component = MockComponent(name)
                        registry.register(name, component)
                        results.append(('register_success', name))
                    except Exception:
                        results.append(('register_failed', name))

                elif operation == 'get':
                    # Try to get an existing component
                    if component_names:
                        name = random.choice(component_names)
                        try:
                            component = registry.get(name)
                            results.append(('get_success', name, component.name))
                        except Exception:
                            results.append(('get_failed', name))

                elif operation == 'has':
                    # Check if component exists
                    if component_names:
                        name = random.choice(component_names)
                        exists = registry.has(name)
                        results.append(('has', name, exists))

                elif operation == 'list':
                    # List all components
                    components = registry.list_components()
                    results.append(('list', len(components)))

                # Small delay
                time.sleep(random.uniform(0.001, 0.003))

            return (thread_id, results)

        # Run mixed operations
        num_threads = 8
        operations_per_thread = 100
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(mixed_operations, i, operations_per_thread)
                for i in range(num_threads)
            ]
            results = [future.result() for future in as_completed(futures)]

        # Basic sanity checks
        total_operations = sum(len(result[1]) for result in results)
        expected_total = num_threads * operations_per_thread
        assert total_operations == expected_total

        # Registry should still be functional
        for name in component_names[:3]:
            component = registry.get(name)
            assert isinstance(component, MockComponent)
            assert component.name == name

    def test_transactional_access(self):
        """Test transactional access to multiple components."""
        registry = ComponentRegistry()
        component_names = ["trans_comp_1", "trans_comp_2", "trans_comp_3"]

        # Register components
        for name in component_names:
            registry.register(name, MockComponent(name))

        def transactional_access(thread_id: int):
            """Access multiple components within a transaction."""
            with registry.transaction() as reg:
                results = []
                for name in component_names:
                    component = reg.get(name)
                    result = component.access()
                    results.append((name, result))
                    # Small delay within transaction
                    time.sleep(0.001)
                return (thread_id, results)

        # Run concurrent transactions
        num_threads = 5
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(transactional_access, i) for i in range(num_threads)]
            results = [future.result() for future in as_completed(futures)]

        # Verify all transactions succeeded
        assert len(results) == num_threads

        for thread_id, access_results in results:
            assert len(access_results) == len(component_names)
            for name, result in access_results:
                assert name in component_names
                assert "accessed" in result

    def test_component_destruction(self):
        """Test thread-safe component destruction."""
        registry = ComponentRegistry()
        component_name = "destructible_component"

        # Create component with destruction tracking
        destruction_called = threading.Event()

        def destructor(component):
            """Mock destructor."""
            destruction_called.set()

        # Register component with destructor
        component = MockComponent(component_name)
        registry.register(component_name, component, on_destroy=destructor)

        def remove_component(thread_id: int):
            """Remove component from thread."""
            try:
                result = registry.remove(component_name)
                return (thread_id, result)
            except Exception as e:
                return (thread_id, False, str(e))

        # Multiple threads trying to remove the same component
        num_threads = 5
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(remove_component, i) for i in range(num_threads)]
            results = [future.result() for future in as_completed(futures)]

        # Only one should succeed
        successful_removals = [r for r in results if r[1]]
        failed_removals = [r for r in results if not r[1]]

        assert len(successful_removals) == 1
        assert len(failed_removals) == num_threads - 1

        # Destructor should have been called
        assert destruction_called.is_set()

        # Component should no longer exist
        with pytest.raises(RuntimeError):
            registry.get(component_name)

    def test_clear_under_concurrent_access(self):
        """Test clearing registry while other threads are accessing it."""
        registry = ComponentRegistry()
        num_components = 10

        # Register components
        component_names = [f"clear_component_{i}" for i in range(num_components)]
        for name in component_names:
            registry.register(name, MockComponent(name))

        clear_called = threading.Event()
        access_results = []

        def clear_registry():
            """Clear the registry."""
            time.sleep(0.05)  # Let some accesses happen first
            registry.clear()
            clear_called.set()

        def access_components(thread_id: int):
            """Continuously access components."""
            while not clear_called.is_set():
                try:
                    name = random.choice(component_names)
                    component = registry.get(name)
                    result = component.access()
                    access_results.append((thread_id, name, result))
                except RuntimeError:
                    # Component might have been cleared
                    access_results.append((thread_id, 'cleared', 'error'))
                time.sleep(0.001)

        # Start threads
        with ThreadPoolExecutor(max_workers=6) as executor:
            clear_future = executor.submit(clear_registry)
            access_futures = [
                executor.submit(access_components, i)
                for i in range(5)
            ]

            # Wait for clear to complete
            clear_future.result()

            # Wait a bit more for access threads to finish
            time.sleep(0.02)

        # Registry should be empty
        assert len(registry.list_components()) == 0

        # Should have some successful accesses before clear
        successful_accesses = [
            r for r in access_results
            if r[1] != 'cleared'
        ]
        assert len(successful_accesses) > 0

    def test_registry_isolation(self):
        """Test that different registries are properly isolated."""
        num_registries = 5
        registries = [ComponentRegistry() for _ in range(num_registries)]
        component_name = "isolated_component"

        def operate_on_registry(registry_id: int, registry: ComponentRegistry):
            """Operate on a specific registry."""
            component = MockComponent(f"{component_name}_{registry_id}")
            registry.register(component_name, component)

            # Access the component
            retrieved = registry.get(component_name)
            return (registry_id, retrieved.name)

        # Operate on different registries concurrently
        with ThreadPoolExecutor(max_workers=num_registries) as executor:
            futures = [
                executor.submit(operate_on_registry, i, registry)
                for i, registry in enumerate(registries)
            ]
            results = [future.result() for future in as_completed(futures)]

        # Each registry should have its own component
        assert len(results) == num_registries
        for registry_id, retrieved_name in results:
            expected_name = f"{component_name}_{registry_id}"
            assert expected_name == retrieved_name, f"Expected {expected_name}, got {retrieved_name}"

        # Verify isolation
        for i, registry in enumerate(registries):
            component = registry.get(component_name)
            expected_name = f"{component_name}_{i}"
            assert component.name == expected_name, f"Registry {i} should have component {expected_name}, got {component.name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])