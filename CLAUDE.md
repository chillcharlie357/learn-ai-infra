# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important: YAML Front Matter Requirement

**When creating or modifying markdown files, you MUST add YAML front matter at the top of the file.**

Example format:
```yaml
---
title: Your Document Title
date: 2026-01-01 23:15    # ← Use current system time
modified: 2026-01-01 23:15 # ← Same as date for new files, update when modifying existing files
tags:
  - tag1
  - tag2
categories:
  - 技术分享
excerpt: Brief description of the document content
mathjax: true  # Set to true if document contains mathematical formulas
comment: true  # Set to true to enable comments
---
```

**Usage guidelines**:
- Use `date '+%Y-%m-%d %H:%M'` command to get current system time
- For new files: `date` and `modified` should be the same
- For existing files: update only the `modified` field
- Choose appropriate tags based on content (e.g., `cuda`, `gpu-architecture`, `parallel-computing`)
- Use categories like: `技术分享`, `学习笔记`, `源码分析`, `实践项目`
- Set `mathjax: true` only when document contains LaTeX/math formulas
- Set `comment: true` to enable discussion on the document

## Repository Purpose

This is a **learning repository for AI infrastructure** (learn-ai-infra), designed to bridge the gap between algorithm engineering and infrastructure engineering. The goal is to master the underlying technologies powering modern AI systems, from GPU architecture to distributed training frameworks.

This is a **knowledge repository**, not a traditional software project with build/test pipelines. Content includes documentation, code examples, implementation notes, and research findings organized along a structured learning path.

## Learning Roadmap Structure

The repository follows a three-tier progression defined in `road-map.md`:

### Tier 1: GPU & Operator Layer (底层进阶)
**Goal**: Master GPU architecture and high-performance computing fundamentals

- **GPU Architecture**: SM (Streaming Multiprocessor), Warp scheduling, Tensor Core acceleration
- **Memory Hierarchy**: Register → Shared Memory → L2 Cache → HBM
- **Performance Analysis**: Roofline Model, compute-bound vs memory-bound operations
- **Core Operators**: LLM operators (Attention, GEMM, Softmax) implementation and optimization
- **Practice**: Implement FlashAttention from scratch

### Tier 2: Framework Layer (系统集成)
**Goal**: Understand full-stack integration from PyTorch to large-scale training/inference frameworks

**General Capabilities**:
- **Parallel Strategies**: DP/ZeRO (Data Parallel), TP/PP (Tensor/Pipeline Parallel), CP/EP (Context/Expert Parallel)
- **Memory Management**: Maximize GPU memory utilization under resource constraints

**Inference Framework Specifics**:
- **KV Cache Management**: PagedAttention for long-context inference
- **Batch Scheduling**: Continuous Batching and dynamic scheduling strategies
- **PD Separation**: Prefill (warmup) vs Decoding phases separation

### Tier 3: Open Source Contribution
**Goal**: Contribute to real-world AI infrastructure projects

**Advanced Path**:
- Contribute to vLLM
- Contribute to SGLang

**Entry Path (simplified frameworks)**:
- Study nano-vLLM source code
- Study mini-sglang source code
- **Action**: Master core inference framework concepts through these simplified implementations

## Documentation Organization

### `/cuda/` Directory
Contains GPU architecture and CUDA programming documentation:

- **`nvidia_gpu_basic_structures.md`**: Comprehensive reference on NVIDIA GPU fundamentals
  - Execution hierarchy: Thread → Warp (32 threads) → Thread Block → Grid
  - Memory hierarchy: Registers → Shared Memory (48KB default, configurable 16-96KB) → Global Memory
  - SIMT architecture and optimization techniques
  - Code examples: Hello World, matrix multiplication
  - Performance optimization strategies

When adding CUDA documentation:
- Include verified technical specifications with sources
- Provide practical code examples
- Reference official NVIDIA documentation and contemporary hardware specs (H100, H200, RTX 4090)
- Focus on concepts relevant to AI workloads (matrix operations, memory access patterns)

### `road-map.md`
The master learning roadmap in Chinese. When adding content, reference the relevant tier and checklist items.

## Content Guidelines

### Adding Documentation
- **YAML Front Matter**: ALWAYS add YAML front matter to new markdown files (see examples above)
- **Structure**: Follow the three-tier roadmap organization
- **Language**: Primary documentation in Chinese, code comments and technical terms in English
- **Code Examples**: Include practical, runnable examples when explaining concepts
- **References**: Cite sources (papers, official docs, blog posts) for technical claims
- **Verification**: When stating hardware specifications or performance numbers, verify against current sources (NVIDIA docs, recent benchmarks)

### Adding Implementation Projects
- Place implementation code in appropriate subdirectories (e.g., `/cuda/` for GPU operators, `/frameworks/` for framework studies)
- Include README explaining implementation approach, key learnings, and performance observations
- Document knowledge gaps discovered during implementation
- Compare with reference implementations when available

### Research Notes
- When studying open-source projects (vLLM, SGLang, nano-vLLM, mini-sglang):
  - Take notes on architecture design choices
  - Document optimization techniques discovered
  - Create diagrams for complex systems (e.g., KV cache management, parallel strategies)
  - Track understanding of core abstractions

## Key Concepts to Maintain

This repository focuses on **infrastructure fundamentals**, not application-level ML. Keep content centered on:

1. **Hardware awareness**: GPU architecture, memory hierarchy, parallel execution
2. **Performance engineering**: Roofline model, profiling, optimization
3. **Systems integration**: How components work together in training/inference pipelines
4. **Practical implementation**: Real code, not just theory

Avoid:
- High-level ML algorithm tutorials (unless demonstrating infrastructure concepts)
- Application-level feature development
- Generic software engineering practices

## Common Workflows

### Studying an Open-Source Framework
1. Create a dedicated directory (e.g., `/studies/nano-vllm/`)
2. Start with architecture overview (README, design docs)
3. Trace core execution paths (e.g., request handling in inference frameworks)
4. Document key components and their interactions
5. Identify and explain optimization techniques
6. Create minimal reproducible examples of interesting patterns

### Implementing GPU Operators
1. Create directory under `/cuda/operators/` (e.g., `flash_attention/`)
2. Start with naive implementation for correctness
3. Profile to identify bottlenecks (compute-bound vs memory-bound)
4. Apply optimizations incrementally (shared memory tiling, register blocking, etc.)
5. Document each optimization with performance measurements
6. Compare against reference implementations (cuDNN, FlashAttention, etc.)

### Contributing to Learning Roadmap
- Update `road-map.md` checklist as items are completed
- Add links to related documentation/experiments in appropriate roadmap sections
- Track progress on understanding vs implementation (knowing how something works vs being able to build it)

## Technology Focus

**Core Areas**:
- CUDA programming and GPU optimization
- PyTorch internals (CUDA extension development, autograd mechanics)
- Distributed training (NCCL, collective communication)
- LLM inference optimization (KV cache, batching, quantization)

**Key Projects to Study**:
- vLLM (inference framework with PagedAttention)
- SGLang (speculative inference, KV cache optimization)
- nano-vLLM / mini-sglang (simplified reference implementations)
- PyTorch (CUDA ops, distributed package)
- DeepSpeed / Megatron-LM (distributed training)

When working with these projects, focus on infrastructure insights rather than usage tutorials.

---

## Git Commit Message Convention

**All git commits in this repository MUST follow the Conventional Commits specification.**

Reference: [Conventional Commits Cheatsheet](https://gist.github.com/qoomon/5dfcdf8eec66a051ecd85625518cfd13)

### Basic Format

```
<type>(<optional scope>): <description>

<optional body>

<optional footer>
```

### Commit Types

Use the following types for categorizing changes:

- **`feat`** - Add, adjust, or remove a new feature to the API or UI
- **`fix`** - Fix an API or UI bug of a preceding `feat` commit
- **`refactor`** - Rewrite or restructure code without altering API or UI behavior
- **`perf`** - Performance improvements (special type of refactor)
- **`style`** - Code style changes (whitespace, formatting, missing semi-colons) that don't affect behavior
- **`test`** - Add missing tests or correct existing ones
- **`docs`** - Documentation-only changes
- **`build`** - Build-related changes (build tools, dependencies, project version)
- **`ops`** - Operational aspects (infrastructure/IaC, deployment scripts, CI/CD, backups, monitoring)
- **`chore`** - Maintenance tasks (initial commit, modifying `.gitignore`, etc.)

### Scope (Optional)

The `scope` provides additional contextual information:
- Optional part of the commit message
- Define project-specific scopes as needed
- Common scopes for this repo: `cuda`, `docs`, `roadmap`, `frameworks`, `gpu-ops`

### Breaking Changes

- Indicate breaking changes with `!` before the `:`: `feat(api)!: remove status endpoint`
- Describe breaking changes in the commit footer if the description isn't sufficient
- Footer must start with `BREAKING CHANGE:`

### Description Rules

The `description` is **mandatory**:
- Use imperative, present tense: "change" not "changed" nor "changes"
- Think: "This commit will..." or "This commit should..."
- **Do not** capitalize the first letter
- **Do not** end with a period (`.`)

### Examples

```bash
# Add new documentation
feat(docs): add GPU memory optimization guide

# Fix a bug
fix(cuda): correct shared memory size specification

# Refactor code
refactor(gpu-ops): simplify matrix multiplication kernel

# Performance improvement
perf: reduce kernel launch overhead

# Update dependencies
build: update CUDA toolkit to version 12.0

# Documentation only
docs: clarify warp execution model

# Breaking change
feat!: remove deprecated memory allocation API

BREAKING CHANGE: The old allocation API is no longer supported.
Use the new unified allocation interface instead.
```

### Decision Tree for Choosing Type

When in doubt, follow this priority order:

1. **Bug fix?** → `fix:`
2. **New or changed feature?** → `feat:`
3. **Performance improvement?** → `perf:`
4. **Code restructuring without behavior change?** → `refactor:`
5. **Tests added/corrected?** → `test:`
6. **Documentation only?** → `docs:`
7. **Code style/formatting only?** → `style:`
8. **Build tools, dependencies?** → `build:`
9. **DevOps/infrastructure/backups?** → `ops:`
10. **Maintenance or non-code task?** → `chore:`

### Multi-Line Commit Example

```bash
git commit -m "feat(cuda): implement tiled matrix multiplication

- Add shared memory tiling for better cache utilization
- Optimize memory access patterns for coalesced reads
- Achieve 2.5x performance improvement on RTX 4090

Ref: https://github.com/username/repo/pull/123"
```

Remember: Conventional commits enable automated changelog generation and semantic versioning. Use them consistently.

---

## Git Commit Helpers

This repository includes custom slash commands that automatically analyze changes and generate Conventional Commits messages:

### 1. Local Commit Only
```
/git-commit
```
- **Automatically analyzes changes** to determine commit type, scope, and description
- Generates a Conventional Commits message following the specification
- Shows the proposed message for review
- User only needs to confirm (or provide custom message)
- Auto-stages all changes if needed
- **Commits to local repository only**

### 2. Commit and Push
```
/git-push
```
- Same auto-analysis as `/git-commit`
- Generates commit message automatically
- User confirms the message
- After committing, automatically pushes to remote
- Detects remote status and shows unpushed commits count
- Sets upstream tracking if needed
- **Commits to local repository AND pushes to remote**

### Usage Examples

```bash
# Make some changes to your files
# Then in Claude Code:

# Commit locally only (auto-generates message)
/git-commit

# OR commit and push in one go (auto-generates message)
/git-push
```

### How Auto-Generation Works

The commands analyze your changes to determine:

**Commit Type:**
- `feat` - New files, new features, implementations
- `fix` - Bug fixes, error handling, corrections
- `refactor` - Code restructuring without behavior change
- `perf` - Performance optimizations
- `style` - Formatting, whitespace changes
- `test` - Test files modified/added
- `docs` - Documentation changes
- `build` - Build configs, dependencies
- `ops` - CI/CD, infrastructure
- `chore` - Maintenance tasks

**Scope Detection:**
- `cuda` - Files in cuda/ directory
- `docs` - Documentation files
- `roadmap` - Changes to roadmap.md
- `frameworks` - Framework-related code
- `gpu-ops` - GPU operator implementations
- `claude` - .claude/ directory changes

**Message Format:**
- Imperative present tense ("add" not "added")
- Lowercase first letter
- No period at end
- Summarizes main changes

### Example Output

```
Analyzing changes...

📝 Proposed commit message:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
feat(cuda): add GPU memory optimization documentation

- Add comprehensive guide on GPU memory hierarchy
- Include verified specifications for modern GPUs
- Add performance optimization strategies
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Proceed with commit? (y/N):
```

If you reject the proposal, you can provide a custom message or abort.
