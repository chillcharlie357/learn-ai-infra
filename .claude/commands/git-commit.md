---
description: Analyze changes and generate Conventional Commits message for local commit
---

Analyze all changes in the repository and automatically generate a Conventional Commits message, then ask user for confirmation before committing.

**Steps:**
1. Check current git status
2. Run `git diff --staged` to see staged changes
3. Run `git diff` to see unstaged changes
4. Analyze changed files and content to determine:
   - Commit type (feat/fix/refactor/perf/style/test/docs/build/ops/chore)
   - Scope (based on directory/file patterns)
   - Description (summary of changes)
5. Generate commit message following Conventional Commits spec
6. Show the proposed commit message and wait for user confirmation
7. If user confirms: stage all changes (if not staged) and commit
8. If user rejects: allow user to provide alternative message or abort

**Auto-detection Rules:**

**Commit Type Detection:**
- `feat` - New files added, new functions/classes, feature implementations
- `fix` - Bug fixes, error handling, corrections
- `refactor` - Code restructuring without behavior change
- `perf` - Performance optimizations, caching, memory improvements
- `style` - Formatting, whitespace, style-only changes
- `test` - Test files modified/added
- `docs` - Documentation changes (*.md files in docs/ or README)
- `build` - Build configs, dependencies, package files
- `ops` - CI/CD, infrastructure, deployment configs
- `chore` - Maintenance, .gitignore, config updates

**Scope Detection:**
- `cuda` - Files in cuda/ directory
- `docs` - Documentation files
- `roadmap` - Changes to roadmap.md
- `frameworks` - Framework-related code
- `gpu-ops` - GPU operator implementations
- `claude` - .claude/ directory changes

**Description Format:**
- Use imperative present tense
- Start with lowercase
- No period at the end
- Summarize the main change

**Breaking Change Detection:**
- Look for "BREAKING", "deprecated", "remove" in changed content
- Check for API signature changes
- Flag if major structural changes detected

**Example Output:**
```
Analyzing changes...

📝 Proposed commit message:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
feat(cuda): add GPU memory optimization documentation

- Add comprehensive guide on GPU memory hierarchy
- Include verified specifications for modern GPUs
- Add performance optimization strategies
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

After showing the proposed message, wait for user to confirm or provide alternative message.
