# Task data provenance

58 Java tasks across four datasets. This file records where each one came
from and under what terms, so that the data can be reused without pulling
in the harness and without a licence surprise.

**Nothing here is a verbatim copy of upstream source code.** Every task is a
self-contained Java class written for this benchmark. Where a task is
derived from an upstream benchmark, what is derived is the *problem* — the
specification, the input/output pairs, the shape of the bug — not the
implementation. The JUnit suites are original.

## Where the tasks live

The 58 tasks are stored in two places, which matters if you want to reuse
them:

| Location | Count | Contents |
|---|---|---|
| `data/*/*.jsonl` | 46 | The bulk of every dataset |
| `bench/datasets/*.py` (`_EXAMPLE_PROBLEMS`) | 12 | 3 per dataset, embedded in the loaders |

Each loader returns built-ins **plus** JSONL, so `len(HumanEvalDataset())` is
23, not 20. Reusing the data outside this repo means extracting both. The
embedded 12 are Apache-2.0 as part of the source files; they are also offered
under CC-BY-4.0 as data.

## Per-dataset provenance

### humaneval-java — 23 tasks

- **Derived from:** HumanEval (Chen et al., 2021, *Evaluating Large Language
  Models Trained on Code*), <https://github.com/openai/human-eval>.
- **Upstream licence:** MIT.
- **What was taken:** problem statements and test input/output pairs, for a
  subset of problems. `metadata.original_index` records the upstream index.
- **What is original:** the Java translation, the method signatures, the
  JUnit 4 suites, and the reference solutions.
- **Redistribution:** MIT permits this with attribution. Keep the upstream
  copyright notice when redistributing derived problem statements.

### mbpp-java — 18 tasks

- **Derived from:** MBPP (Austin et al., 2021, *Program Synthesis with Large
  Language Models*), <https://github.com/google-research/google-research/tree/master/mbpp>.
- **Upstream licence:** CC-BY-4.0.
- **What was taken:** natural-language task descriptions and test cases for a
  subset of problems. `metadata.original_index` records the upstream index.
- **What is original:** the Java translation and the JUnit 4 suites.
- **Redistribution:** CC-BY-4.0 requires attribution and licence notice.
  This directory's CC-BY-4.0 licence is compatible.

### defects4j — 10 tasks

- **Inspired by:** Defects4J (Just, Jalali & Ernst, 2014), <https://github.com/rjust/defects4j>.
  The Defects4J *framework* is MIT; the bugs it indexes live in projects
  under a mix of Apache-2.0 (Commons Lang, Commons Math, Closure Compiler,
  Joda-Time), LGPL-2.1 (JFreeChart) and MIT (Mockito).
- **What was taken:** the *description* of a real defect — which method, what
  went wrong, which test caught it. `metadata.project`, `metadata.bug_id`,
  `metadata.buggy_file` and `metadata.failing_test` name the upstream
  defect so a reader can look it up.
- **What is original: the code.** Each task is a 10–30 line standalone class
  written to exhibit the same defect. No upstream file was copied, in whole
  or in part. The largest task is 27 lines; the upstream files it references
  run to thousands.
- **Why this matters:** JFreeChart is LGPL. Had upstream source been copied,
  `D4J_Chart_3` and `D4J_Chart_9` could not be relicensed, and the copyleft
  would reach anything distributing them. Because only the defect description
  was reused, no upstream copyright subsists in these files. Defect
  descriptions and file paths are facts about a codebase, not expression.
- **If you extend this dataset:** do not paste upstream source in. Reconstruct
  the defect. If you must use real source, move the affected tasks into a
  separate directory carrying the upstream licence, as this file's structure
  anticipates.

### godclass — 7 tasks

- **Origin:** written for this project, from the author's thesis on God Class
  detection and decomposition. `metadata.source_project` is `thesis-example`
  for all seven.
- **Upstream licence:** none. Sole author, no third-party code.
- **Metadata:** `original_loc`, `original_wmc`, `original_cbo`, `original_lcom`,
  `original_rfc` are the CK metrics of the class before decomposition, and
  `severity` is the 4-level classification (`moderate`, `severe`, `extreme`).

## Contamination

Every task predates the training cutoff of every model evaluated in the
published run, and HumanEval and MBPP in particular are near-certainly in the
training data of all three. Results on those two datasets should be read as an
upper bound. See the Limitations section of the top-level README.

## Attribution

When redistributing this data, cite this repository (see `../CITATION.cff`)
and the upstream benchmarks named above.
