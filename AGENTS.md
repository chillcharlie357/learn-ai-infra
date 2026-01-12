# AGENTS.md

This file provides guidance for AI coding agents working in this repository.

## Repository Type

This is a **knowledge repository** for AI infrastructure learning - a documentation-focused repo, not a traditional software project. Content includes markdown documentation, code examples, and research notes organized along a structured learning path.

## Build/Lint/Test Commands

**None** - This repository has no build system, linting tools, or test suite. No package.json, Makefile, or CI pipelines exist.

## Documentation Guidelines

### YAML Front Matter (REQUIRED)

ALL markdown files MUST include YAML front matter:

```yaml
---
title: Your Document Title
date: 2026-01-01 23:15    # Use current system time: date '+%Y-%m-%d %H:%M'
modified: 2026-01-01 23:15 # Same as date for new files, update only when modifying
tags:
  - cuda
  - gpu-architecture
categories:
  - 技术分享
excerpt: Brief description
mathjax: true  # Set only if document contains LaTeX/math formulas
comment: true  # Set to enable comments
---
```

### Data Verification (CRITICAL)

All technical data MUST be verified and cited:

- Hardware specs (bandwidth, memory, compute capability)
- Performance numbers (TFLOPS, latency, benchmarks)
- Architecture details and optimization techniques
- Any quantitative information

**Citation format:**
- Use numbered footnotes: [¹], [²], [³]
- Add "数据来源" or "Data Sources" section at document end
- Format: `- **[¹] [Source Title](URL)** - Brief description`

**Verification sources:** Official documentation, peer-reviewed papers, reputable tech blogs (prefer recent, within 2-3 years)

### Structure & Style

- **Language**: Primary text in Chinese, code comments/technical terms in English
- **Organization**: Follow three-tier roadmap (GPU & Operator → Framework → Open Source)
- **Code Examples**: Include practical, runnable examples
- **Focus**: Infrastructure fundamentals (GPU architecture, performance engineering, systems integration)

## Git Commit Convention

Follow Conventional Commits specification:

```
<type>(<optional scope>): <description>
```

**Types:**
- `feat` - New features/implementation
- `fix` - Bug fixes/corrections
- `refactor` - Restructuring without behavior change
- `perf` - Performance improvements
- `docs` - Documentation-only changes
- `style` - Formatting/whitespace
- `test` - Test additions/corrections
- `build` - Build tools/dependencies
- `ops` - CI/CD/infrastructure
- `chore` - Maintenance

**Common scopes:** `cuda`, `docs`, `roadmap`, `frameworks`, `gpu-ops`

**Description rules:**
- Imperative present tense: "add" not "added"
- Lowercase first letter
- No period at end

**Breaking changes:** Add `!` before `:` and include `BREAKING CHANGE:` footer

### Slash Commands

Use `/git-commit` for local commit, `/git-push` for commit+push. Both auto-analyze changes and generate Conventional Commits messages.

## What This Repository Focuses On

**Core areas:**
- CUDA programming and GPU optimization
- PyTorch internals (CUDA extensions, autograd)
- Distributed training (NCCL, collective communication)
- LLM inference optimization (KV cache, batching, quantization)

**Avoid:**
- High-level ML algorithm tutorials (unless demonstrating infrastructure)
- Application-level feature development
- Generic software engineering practices

## File Organization

- `/cuda/` - GPU architecture and CUDA programming documentation
- `road-map.md` - Master learning roadmap (Chinese)
- `CLAUDE.md` - Detailed project guidelines (this is the canonical reference)
- `README.md` - Repository overview

## When Adding Content

1. **Documentation**: Add appropriate YAML front matter, verify/cite all data
2. **Implementation projects**: Place in subdirectories with README explaining approach/learnings
3. **Research notes**: Document architecture choices, optimizations, create diagrams for complex systems
4. **Roadmap**: Update road-map.md checklist as items completed
