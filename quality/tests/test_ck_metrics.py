"""
test_ck_metrics.py — Thorough tests for the CK metrics extractor.

Tests cover:
    - LOC counting (blank lines, comments, block comments)
    - WMC computation (simple methods, complex methods, constructors)
    - DIT computation (single inheritance, chain, Object base)
    - NOC computation (parent-child relationships)
    - CBO computation (external type references, stdlib exclusion)
    - RFC computation (own methods + called methods)
    - LCOM computation (cohesion, shared fields, disjoint methods)
    - Multi-class files
    - Interface handling
    - Malformed / unparseable input
    - Real-world samples (fixture files)
"""

import sys
from pathlib import Path

import pytest

# Ensure quality package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quality.ck_metrics import (
    CKMetricsExtractor,
    _resolve_dit,
    count_logical_loc,
    extract_from_file,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ===================================================================
# LOC counting
# ===================================================================

class TestLogicalLOC:

    def test_empty_string(self):
        assert count_logical_loc("") == 0

    def test_blank_lines_only(self):
        assert count_logical_loc("\n\n\n") == 0

    def test_simple_code(self):
        src = "int x = 1;\nint y = 2;\nreturn x + y;"
        assert count_logical_loc(src) == 3

    def test_single_line_comments_excluded(self):
        src = "// comment\nint x = 1;\n// another\nreturn x;"
        assert count_logical_loc(src) == 2

    def test_block_comments_excluded(self):
        src = "/* start\n * middle\n */\nint x = 1;"
        assert count_logical_loc(src) == 1

    def test_javadoc_excluded(self):
        src = "/** Javadoc.\n * @param x desc\n */\npublic void foo() {}"
        assert count_logical_loc(src) == 1

    def test_mixed_code_and_comments(self):
        src = (
            "package com.example;\n"
            "\n"
            "// imports\n"
            "import java.util.List;\n"
            "\n"
            "/**\n"
            " * Class doc.\n"
            " */\n"
            "public class Foo {\n"
            "    int x;\n"
            "}\n"
        )
        # Lines: package, import, class, int x, } = 5
        assert count_logical_loc(src) == 5

    def test_inline_block_comment(self):
        src = "int x = /* value */ 42;"
        # This counts as 1 line since /* */ is on same line
        assert count_logical_loc(src) == 1


# ===================================================================
# WMC (Weighted Methods per Class)
# ===================================================================

class TestWMC:

    def test_empty_class(self):
        src = "public class Empty {}"
        extractor = CKMetricsExtractor()
        metrics = extractor.extract(src)
        assert len(metrics) == 1
        assert metrics[0].wmc == 0
        assert metrics[0].num_methods == 0

    def test_single_simple_method(self):
        src = """
        public class Simple {
            public int getValue() {
                return 42;
            }
        }
        """
        metrics = CKMetricsExtractor().extract(src)
        assert metrics[0].wmc == 1  # CC=1 for a simple method
        assert metrics[0].num_methods == 1

    def test_method_with_if(self):
        src = """
        public class WithIf {
            public int check(int x) {
                if (x > 0) {
                    return x;
                } else {
                    return -x;
                }
            }
        }
        """
        metrics = CKMetricsExtractor().extract(src)
        assert metrics[0].wmc == 2  # CC=2 (1 base + 1 if)

    def test_method_with_loop_and_conditions(self):
        src = """
        public class Complex {
            public int process(int[] data) {
                int sum = 0;
                for (int val : data) {
                    if (val > 0) {
                        sum += val;
                    } else if (val == 0) {
                        continue;
                    }
                }
                return sum;
            }
        }
        """
        metrics = CKMetricsExtractor().extract(src)
        # CC: 1 base + 1 for + 1 if + 1 else-if = 4
        assert metrics[0].wmc >= 3

    def test_multiple_methods_summed(self):
        src = """
        public class Multi {
            public int a() { return 1; }
            public int b(int x) {
                if (x > 0) return x;
                return -x;
            }
            public void c() {}
        }
        """
        metrics = CKMetricsExtractor().extract(src)
        assert metrics[0].num_methods == 3
        # a=1, b=2, c=1 => WMC=4
        assert metrics[0].wmc == 4

    def test_constructor_counted(self):
        src = """
        public class WithCtor {
            private int x;
            public WithCtor(int x) {
                if (x < 0) throw new IllegalArgumentException();
                this.x = x;
            }
        }
        """
        metrics = CKMetricsExtractor().extract(src)
        assert metrics[0].wmc >= 2  # constructor with if


# ===================================================================
# DIT (Depth of Inheritance Tree)
# ===================================================================

class TestDIT:

    def test_no_extends(self):
        src = "public class Base {}"
        metrics = CKMetricsExtractor().extract(src)
        assert metrics[0].dit == 0

    def test_extends_object(self):
        src = "public class Base extends Object {}"
        metrics = CKMetricsExtractor().extract(src)
        assert metrics[0].dit == 0

    def test_single_inheritance(self):
        src = """
        public class Parent {}
        public class Child extends Parent {}
        """
        metrics = CKMetricsExtractor().extract(src)
        parent = next(m for m in metrics if m.class_name == "Parent")
        child = next(m for m in metrics if m.class_name == "Child")
        assert parent.dit == 0
        assert child.dit == 1

    def test_inheritance_chain(self):
        src = """
        public class A {}
        public class B extends A {}
        public class C extends B {}
        """
        metrics = CKMetricsExtractor().extract(src)
        c = next(m for m in metrics if m.class_name == "C")
        assert c.dit == 2

    def test_resolve_dit_with_cache(self):
        parent_map = {"A": None, "B": "A", "C": "B", "D": "C"}
        cache = {}
        assert _resolve_dit("D", parent_map, cache) == 3
        assert cache["D"] == 3
        assert cache["C"] == 2
        assert cache["B"] == 1


# ===================================================================
# NOC (Number of Children)
# ===================================================================

class TestNOC:

    def test_no_children(self):
        src = "public class Leaf {}"
        metrics = CKMetricsExtractor().extract(src)
        assert metrics[0].noc == 0

    def test_one_child(self):
        src = """
        public class Parent {}
        public class Child extends Parent {}
        """
        metrics = CKMetricsExtractor().extract(src)
        parent = next(m for m in metrics if m.class_name == "Parent")
        assert parent.noc == 1

    def test_multiple_children(self):
        src = """
        public class Animal {}
        public class Dog extends Animal {}
        public class Cat extends Animal {}
        public class Bird extends Animal {}
        """
        metrics = CKMetricsExtractor().extract(src)
        animal = next(m for m in metrics if m.class_name == "Animal")
        assert animal.noc == 3


# ===================================================================
# CBO (Coupling Between Objects)
# ===================================================================

class TestCBO:

    def test_no_coupling(self):
        src = """
        public class Isolated {
            private int x;
            public int getX() { return x; }
        }
        """
        metrics = CKMetricsExtractor().extract(src)
        assert metrics[0].cbo == 0

    def test_stdlib_excluded_by_default(self):
        src = """
        import java.util.List;
        import java.util.ArrayList;
        public class UsesStdlib {
            private List<String> items = new ArrayList<>();
        }
        """
        metrics = CKMetricsExtractor().extract(src)
        assert metrics[0].cbo == 0

    def test_stdlib_included_when_requested(self):
        src = """
        import java.util.List;
        import java.util.ArrayList;
        public class UsesStdlib {
            private List<String> items;
            public void init() {
                items = new ArrayList<>();
            }
            public void process(StringBuilder sb) {}
        }
        """
        extractor = CKMetricsExtractor(include_stdlib_coupling=True)
        metrics = extractor.extract(src)
        # Should include ArrayList and/or StringBuilder
        assert metrics[0].cbo >= 1

    def test_external_type_counted(self):
        src = """
        public class Service {
            private DatabaseClient db;
            private CacheManager cache;
            public void process(RequestHandler handler) {}
        }
        """
        metrics = CKMetricsExtractor().extract(src)
        # DatabaseClient, CacheManager, RequestHandler = 3
        assert metrics[0].cbo == 3


# ===================================================================
# RFC (Response For a Class)
# ===================================================================

class TestRFC:

    def test_no_methods(self):
        src = "public class Empty {}"
        metrics = CKMetricsExtractor().extract(src)
        assert metrics[0].rfc == 0

    def test_methods_only(self):
        src = """
        public class Simple {
            public void a() {}
            public void b() {}
        }
        """
        metrics = CKMetricsExtractor().extract(src)
        assert metrics[0].rfc == 2

    def test_methods_calling_external(self):
        src = """
        public class Caller {
            public void doWork() {
                System.out.println("hello");
                process();
            }
            private void process() {}
        }
        """
        metrics = CKMetricsExtractor().extract(src)
        # doWork, process are own methods; println is called
        assert metrics[0].rfc >= 3


# ===================================================================
# LCOM (Lack of Cohesion of Methods)
# ===================================================================

class TestLCOM:

    def test_single_method(self):
        src = """
        public class One {
            private int x;
            public int getX() { return x; }
        }
        """
        metrics = CKMetricsExtractor().extract(src)
        assert metrics[0].lcom == 0  # need ≥2 methods

    def test_fully_cohesive(self):
        """Two methods both accessing the same field → LCOM = 0."""
        src = """
        public class Cohesive {
            private int value;
            public int getValue() { return value; }
            public void setValue(int v) { this.value = v; }
        }
        """
        metrics = CKMetricsExtractor().extract(src)
        assert metrics[0].lcom == 0

    def test_no_shared_fields(self):
        """Two methods accessing disjoint fields → LCOM > 0."""
        src = """
        public class Disjoint {
            private int x;
            private int y;
            public int getX() { return x; }
            public int getY() { return y; }
        }
        """
        metrics = CKMetricsExtractor().extract(src)
        assert metrics[0].lcom >= 1

    def test_no_fields(self):
        src = """
        public class NoFields {
            public void a() {}
            public void b() {}
        }
        """
        metrics = CKMetricsExtractor().extract(src)
        assert metrics[0].lcom == 0  # 0 fields → fallback


# ===================================================================
# Multi-class and interface handling
# ===================================================================

class TestMultiClassAndInterface:

    def test_two_classes_in_file(self):
        src = """
        public class ClassA {
            public void foo() {}
        }
        class ClassB {
            public void bar() {}
        }
        """
        metrics = CKMetricsExtractor().extract(src)
        assert len(metrics) == 2
        names = {m.class_name for m in metrics}
        assert names == {"ClassA", "ClassB"}

    def test_interface_extracted(self):
        src = """
        public interface Doable {
            void doIt();
        }
        """
        metrics = CKMetricsExtractor().extract(src)
        assert len(metrics) == 1
        assert metrics[0].class_name == "Doable"


# ===================================================================
# Error handling
# ===================================================================

class TestErrorHandling:

    def test_unparseable_source(self):
        src = "this is not valid java {{{}}}"
        metrics = CKMetricsExtractor().extract(src)
        assert metrics == []

    def test_empty_source(self):
        metrics = CKMetricsExtractor().extract("")
        assert metrics == []

    def test_nonexistent_file(self):
        result = extract_from_file("/nonexistent/path/Fake.java")
        assert result == []


# ===================================================================
# Fixture files
# ===================================================================

class TestFixtureFiles:

    @pytest.fixture(autouse=True)
    def check_fixtures(self):
        if not FIXTURES.exists():
            pytest.skip("Fixture files not available")

    def test_high_quality_file(self):
        metrics = extract_from_file(FIXTURES / "HighQuality.java")
        assert len(metrics) == 1
        m = metrics[0]
        assert m.class_name == "StudentRegistry"
        assert m.loc > 0
        assert m.wmc > 0
        # High quality → expect moderate WMC
        assert m.wmc <= 30

    def test_medium_quality_file(self):
        metrics = extract_from_file(FIXTURES / "MediumQuality.java")
        assert len(metrics) == 1
        m = metrics[0]
        assert m.class_name == "Cart"
        assert m.num_methods >= 5

    def test_low_quality_file(self):
        metrics = extract_from_file(FIXTURES / "LowQuality.java")
        assert len(metrics) == 1
        m = metrics[0]
        assert m.class_name == "DataProcessor"
        # God class → high WMC
        assert m.wmc > 10
        # Many fields
        assert m.num_fields >= 8

    def test_low_quality_has_higher_wmc_than_high(self):
        high = extract_from_file(FIXTURES / "HighQuality.java")[0]
        low = extract_from_file(FIXTURES / "LowQuality.java")[0]
        assert low.wmc > high.wmc

    def test_low_quality_has_higher_lcom(self):
        high = extract_from_file(FIXTURES / "HighQuality.java")[0]
        low = extract_from_file(FIXTURES / "LowQuality.java")[0]
        # Low quality should have worse cohesion
        assert low.lcom >= high.lcom
