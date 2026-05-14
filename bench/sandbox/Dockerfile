# llm-se-bench sandbox — Java compilation and test execution environment
# Supports JDK 17 (default) with JUnit 4 for test execution.
#
# Build:
#   docker build -t llm-se-bench-sandbox:17 .
#
# Usage (by DockerSandbox):
#   docker run --rm --network=none --memory=256m --cpus=1 \
#     -v /path/to/staging:/workspace:rw \
#     llm-se-bench-sandbox:17 \
#     bash -c "cd /workspace && javac -cp /usr/share/java/junit4.jar:. ..."

FROM eclipse-temurin:17-jdk-jammy

LABEL maintainer="llm-se-bench"
LABEL description="Sandbox for compiling and testing LLM-generated Java code"

# Install JUnit 4 and basic build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    junit4 \
    maven \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for sandbox isolation
RUN useradd -m -s /bin/bash sandbox && \
    mkdir -p /workspace && \
    chown sandbox:sandbox /workspace

# Set up classpath
ENV CLASSPATH="/usr/share/java/junit4.jar:/workspace/out:."

WORKDIR /workspace
USER sandbox

# Default: compile and run
CMD ["bash"]
