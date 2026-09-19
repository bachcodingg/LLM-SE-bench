"""
Shared fixtures for the MCP server tests.

Every test here runs offline: no API call, no Docker requirement (the
sandbox degrades to a structural check), and no network.
"""

from __future__ import annotations

import pytest

# A small, cohesive class: two fields, two clusters of methods.
SIMPLE_CLASS = """\
public class Counter {
    private int count;

    public void increment() {
        count = count + 1;
    }

    public int getCount() {
        return count;
    }
}
"""

# Two independent responsibilities sharing no field: the shape
# field-cluster decomposition is meant to find.
TWO_CLUSTER_CLASS = """\
public class OrderProcessor {
    private java.util.List<String> orders;
    private double taxRate;
    private String reportHeader;
    private int reportCount;

    public void addOrder(String order) {
        if (order != null) {
            orders.add(order);
        }
    }

    public double computeTotal(double amount) {
        if (amount < 0) {
            return 0;
        }
        return amount * orders.size() * (1 + taxRate);
    }

    public int orderCount() {
        return orders.size();
    }

    public String buildReport() {
        reportCount = reportCount + 1;
        return reportHeader + " #" + reportCount;
    }

    public void setReportHeader(String header) {
        reportHeader = header;
    }
}
"""

# Every threshold breached several times over.
GOD_CLASS = """\
public class Everything {
    private int a; private int b; private int c; private int d;
    private String e; private String f;

""" + "\n".join(
    f"""    public int method{i}(int x) {{
        if (x > {i}) {{ a = a + 1; }} else if (x < -{i}) {{ b = b - 1; }}
        for (int k = 0; k < x; k++) {{ c += k; }}
        while (d > 0) {{ d--; }}
        java.io.File file{i} = new java.io.File("p{i}");
        java.util.Date date{i} = new java.util.Date();
        Helper{i} helper = new Helper{i}();
        e = helper.describe{i}(e);
        f = helper.render{i}(f);
        return helper.use{i}(a + b + c + d + file{i}.hashCode() + date{i}.hashCode());
    }}"""
    for i in range(20)
) + "\n}\n"

VALID_JUNIT = """\
import org.junit.Test;
import static org.junit.Assert.*;

public class CounterTest {
    @Test
    public void testIncrement() {
        Counter counter = new Counter();
        counter.increment();
        assertEquals(1, counter.getCount());
    }
}
"""

BROKEN_CLASS = """\
public class Broken {
    public int oops( {
        return
}
"""


@pytest.fixture
def simple_class() -> str:
    return SIMPLE_CLASS


@pytest.fixture
def two_cluster_class() -> str:
    return TWO_CLUSTER_CLASS


@pytest.fixture
def god_class() -> str:
    return GOD_CLASS


@pytest.fixture
def valid_junit() -> str:
    return VALID_JUNIT


@pytest.fixture
def broken_class() -> str:
    return BROKEN_CLASS


@pytest.fixture
def java_dir(tmp_path, simple_class, two_cluster_class):
    """A directory holding two parseable Java files."""
    (tmp_path / "Counter.java").write_text(simple_class, encoding="utf-8")
    (tmp_path / "OrderProcessor.java").write_text(two_cluster_class, encoding="utf-8")
    return tmp_path
