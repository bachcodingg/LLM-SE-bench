"""
bench — Benchmark Engine (Component 2) for llm-se-bench.

Manages four evaluation datasets (HumanEval-Java, MBPP-Java, Defects4J,
God Class) and orchestrates end-to-end evaluation: loading problems,
dispatching prompts to the LLM Gateway (C1), executing generated code in
a sandboxed Java environment, running test suites, and recording results.

Public API
----------
Datasets:
    HumanEvalDataset, MBPPDataset, Defects4JDataset, GodClassDataset

Orchestration:
    BenchmarkOrchestrator

Results:
    ResultCollector, PassAtKCalculator

Progress:
    ProgressTracker, CheckpointManager

Sandbox:
    DockerSandbox, SandboxResult

Mock (for testing without C1):
    MockLLMClient
"""

from bench.datasets.defects4j import Defects4JDataset
from bench.datasets.godclass import GodClassDataset
from bench.datasets.humaneval import HumanEvalDataset
from bench.datasets.mbpp import MBPPDataset
from bench.orchestrator import BenchmarkOrchestrator, MockLLMClient
from bench.progress import CheckpointManager, ProgressTracker
from bench.results import PassAtKCalculator, ResultCollector
from bench.sandbox.docker_sandbox import DockerSandbox, SandboxResult

__all__ = [
    "HumanEvalDataset",
    "MBPPDataset",
    "Defects4JDataset",
    "GodClassDataset",
    "BenchmarkOrchestrator",
    "MockLLMClient",
    "ResultCollector",
    "PassAtKCalculator",
    "ProgressTracker",
    "CheckpointManager",
    "DockerSandbox",
    "SandboxResult",
]

__version__ = "0.1.0"
