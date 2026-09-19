"""
taskfactory — mining, building and validating JVM repository tasks (M3).

The hardest module in the v2 plan, and the one with the clearest value
outside this project, because JVM environment setup is a documented
field-wide failure point: EnvBench found the best automated setup method
succeeded on only 29.5% of JVM repositories. Most failures happen before any
code is written, which means most of what a Java benchmark measures is the
harness, not the model.

Pipeline
--------
::

    mine        find merged PRs that change code and tests together
    build       apply the inverse of the fix to create the task
    derive      work out FAIL_TO_PASS and PASS_TO_FAIL
    environment detect the build system, infer the JDK, prewarm dependencies
    validate    prove the task is solvable, deterministic and not flaky
    admit       or reject, with the reason recorded

Nothing is admitted without passing every validation gate. A benchmark is
only as good as its worst task, and a flaky task that fails 30% of the time
for environmental reasons looks exactly like a model capability difference.

Modules
-------
``taskfactory.mining``      GitHub search and PR filtering
``taskfactory.patches``     reverse-patch construction, test-set derivation
``taskfactory.environment`` build system detection, JDK inference, hermetic images
``taskfactory.validation``  the admission gates, including the flake check
``taskfactory.contamination`` commit date versus training cutoff

What is here and what is not
----------------------------
The machinery is here, with tests over fixtures. **Actually mining 150-300
instances is not**: it needs network access, a GitHub token, and many hours
of Maven and Gradle runs against real repositories. ``taskfactory.pipeline``
is the entry point for that, and running it is the user's step, not a thing
that can be faked into existence here.
"""

from __future__ import annotations

__all__ = ["__version__", "TARGET_INSTANCES"]

__version__ = "0.1.0"

#: The target the v2 plan sets. Exceeds SWE-bench-java's 91 and
#: SWE-PolyBench's 165 Java instances, which is the point: Java is thin and
#: badly served, and that is the gap.
TARGET_INSTANCES = (150, 300)
