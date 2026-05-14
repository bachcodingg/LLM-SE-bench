# llm-se-bench

**llm-se-bench** is a reproducible evaluation framework for benchmarking large language models (LLMs) on software engineering tasks. It covers code generation, bug fixing, and refactoring across four Java datasets, and produces statistically rigorous reports with an interactive decision dashboard.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        llm-se-bench pipeline                        │
├──────────┬──────────┬──────────┬──────────────┬─────────────────────┤
│    C1    │    C2    │    C3    │      C4      │         C5          │
│  LLM     │ Bench-   │ Quality  │  Statistical │ Decision            │
│ Gateway  │  mark    │ Analyser │   Engine     │ Framework           │
│          │  Engine  │          │              │                     │
│ • Claude │ • Human  │ • CK     │ • Hypothesis │ • MCDA matrix       │
│ • GPT-4  │   Eval   │   metrics│   tests      │ • Recommender       │
│ • Gemini │ • MBPP   │ • Cyclo- │ • Effect     │ • Pareto frontier   │
│          │ • D4J    │   matic  │   sizes      │ • Dashboard (Dash)  │
│ • Cache  │ • God-   │ • Read-  │ • Bootstrap  │ • PDF / JSON / CSV  │
│ • Cost   │   Class  │   ability│   CIs        │   export            │
│ • Rate   │          │ • Docker │ • Cost model │                     │
│   limit  │          │   sandbox│              │                     │
└──────────┴──────────┴──────────┴──────────────┴─────────────────────┘
       ↓           ↓           ↓           ↓               ↓
   LLMResponse EvalResult QualityMet. StatSummary DecisionMatrix
```

## Installation

```bash
git clone <repo-url>
cd llm-se-bench
pip install -e ".[dev]"
```

Set API keys for the providers you want to use:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
```

## Quick Start

```bash
# Validate datasets
llm-se-bench datasets validate --dataset all

# Run benchmark (dry-run, no real API calls)
llm-se-bench run --dataset humaneval --models claude-3-5-sonnet-20241022 --dry-run

# Run full statistical analysis
llm-se-bench stats --input results/ --quality quality/ --output analysis/

# Generate a decision report
llm-se-bench report --input analysis/statistical_summary.json --format text --profile devops

# Launch the interactive dashboard
llm-se-bench dashboard --port 8050

# Run everything at once from a config file
llm-se-bench full --config pipeline.yaml
```

## Components

### C1 — LLM Gateway (`llm_gateway/`)

Unified client for Claude, GPT-4, and Gemini with response caching (SQLite), cost tracking, rate limiting, and Jinja2 prompt templates.

### C2 — Benchmark Engine (`bench/`)

Evaluates LLMs against HumanEval-Java, MBPP-Java, Defects4J, and God Class datasets. Runs generated Java code in a DockerSandbox, records `EvaluationResult` objects, and computes pass@k.

### C3 — Quality Analyser (`quality/`)

Extracts Chidamber–Kemerer (CK) metrics, cyclomatic complexity, Halstead volume, and readability scores from generated Java code using `javalang` AST parsing.

### C4 — Statistical Engine (`stats/`)

Runs Friedman tests, Nemenyi post-hoc tests, Wilcoxon signed-rank tests, Cliff's delta effect sizes, bootstrap confidence intervals, cost modelling, and consistency analysis. Generates 25+ publication-quality figures and LaTeX tables.

### C5 — Decision Framework (`framework/`)

Multi-Criteria Decision Analysis (MCDA) with configurable weighting profiles (DevOps, Audit, Budget). Produces ranked decision matrices, Pareto frontier analysis, constraint-based model recommendations, an interactive Plotly Dash dashboard, and PDF/JSON/CSV reports.

## Running Tests

```bash
pytest tests/ -v
pytest llm_gateway/tests/ bench/tests/ quality/tests/ stats/tests/ framework/tests/ -v
```

## License

MIT License. See [LICENSE](LICENSE) for details.
