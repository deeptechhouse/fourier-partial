# CLAUDE.md — Andy's AI Projects Workspace (User-Global Conventions)

> **Scope:** Canonical, reusable conventions for all of Andy's work. This file (user-global memory) loads in every session. Workspace-specific context — the project navigation map — lives in project memory at `_andy_ai_projects_2026/CLAUDE.md`. Each rule lives in exactly one place: conventions here, the map there.
> **Last updated:** 2026-05-29

---

## 0. Rule Application Policy (Meta-Rule)

**Core Principle:** Only apply rules directly relevant to the current task. Do not apply coding standards to non-coding tasks, git workflows to non-git operations, or security rules to local file management.

### Task Type → Applicable Rules

| Task Type | Applicable Rules |
|---|---|
| **Code writing/editing** | Code standards, OOP, testing, linting, security, documentation |
| **Planning/architecture** | Planning, design docs, licensing |
| **Git operations** | Git workflow only |
| **File downloads/installs** | Simple commands, local file management |
| **Research/information** | Research rules only |
| **Local file organization** | Local file management, directory index |
| **Non-code documentation** | File output conventions, documentation standards |
| **AI agent development** | AI security (§36), MCP security (§37), code standards, security, internet-facing (§27) |
| **MCP server config** | MCP server security (§37), simple commands |

**Summary output:** Only mention rules actually applied. Silence indicates non-application.

---

## 1. File Types & Output Conventions

- **Code files:** Language-appropriate extension per project.
- **Documentation, design docs, summaries, prompts, peripheral files:** `.html` (browser-readable).
- **Technical Design Documents:** Stylishly formatted `.html`.
- **Error/bug logs:** `.html` (actively updated/appended).
- **Directory indexes:** `NAVIGATION.md` is the primary map (§26); `dir_index.html` is optional and auto-generated, for human-browsed asset folders only (see §13).
- **Next-steps prompts:** `.html` context files + `.txt` terminal command file.

---

## 2. Licensing & Cost Awareness

Before starting any project, **ask whether the project will be sold or available free**.

| Project Type | Preference |
|---|---|
| **Free / Open Source** | Prioritize fully open-source services, libraries, and code. |
| **Commercial** | Use open-source or commercially-permissive code/services (no licensing fees) as much as possible. |

- When a paid service/API is considered: research and display its cost, present alternatives, and prompt the user before committing.
- More options are better than one option.

---

## 3. Planning & Options

**Trigger:** "Create a plan" = produce an optimized, efficient implementation plan following industry best practices.

1. **Research & analyze** — Understand problem space, constraints, and prior art before proposing anything.
2. **Identify best practices** — Established patterns, tools, and approaches that maximize quality and minimize waste.
3. **Optimize for efficiency** — Eliminate unnecessary steps, prefer proven solutions over novel ones, reuse existing code/infrastructure, sequence work for parallel effort.
4. **Present viable options** — Multiple approaches when meaningful trade-offs exist. Ensure cross-step compatibility (e.g., database chosen in step 2 works with framework in step 1).
5. **Structure the plan** — Every plan must include:
   - **Goal** — What success looks like
   - **Approach** — Architecture/strategy with rationale
   - **Steps** — Ordered implementation steps with dependencies noted
   - **Trade-offs** — What was considered and why it was rejected
6. **Optional sections** (include when relevant): Technical architecture, implementation schedule, team/staffing, business/economic aspects.

---

## 4. Technical Design Document

- Output as a **stylishly formatted `.html` file**.
- Cover architecture, data models, API design, service dependencies, and deployment strategy.
- Reflect all choices confirmed during planning.

---

## 5. Code Implementation Standards

- **OOP first:** Object-oriented design principles throughout.
- **Modular design:** Easily updated, swapped, or extended components.
- **Language preference:** Best-practice language for the task. When Java or Python are equivalent, **prefer Java or Python**.
- **No vendor lock-in:** All services/APIs abstracted behind interfaces for easy swapping.

---

## 6. Services & APIs

- **Development:** Prioritize fully functioning **free** services/APIs.
- **Production-ready:** Architected to switch to affordable, reliable paid services upon commercial launch.
- **Paid services acceptable** — research cost, display options, prompt user first.
- **Service abstraction:** Wrap all external calls behind interfaces/adapters so swapping providers requires changing only the adapter.

---

## 7. Graphic Design & GUI

- **Design reference:** Analyze `design_reference/` for each project and make appropriate style choices.
- **Unique per project:** Each project gets its own motif/style/scheme across GUI, website, docs, technical docs, marketing.
- **Editable design:** Config/theme files for colors, fonts, spacing, logos — easily edited post-creation.

---

## 7b. Best Practices Reference

- **Library:** `best_practices_reference/` — software development, security, AI/LLM, audio/MIDI, testing, deployment, language-specific references, tooling.
- **Structure:**
  - `2026_software_development_best_practices.html` — Comprehensive overview
  - `2026-03-13/` — Detailed topic files: web/motion design, AI/LLM, security/compliance, database/OOP, testing/deployment, programming languages, audio/MIDI, RAG, **AI sandboxing** (`report_ai_sandboxing_2026.html`), plus language-specific references (Java, Python, C/C++, C#, Swift, JS/TS, SQL, Assembly/WASM, HTML/CSS) and security tool manuals (Kali, Metasploit, FFmpeg, Argus, Python sockets)
- **Priority:** When these files conflict with general knowledge, prefer the curated references — they reflect the user's vetted, current standards.

---

## 8. Code Verification (Post-Completion)

After code is written:

1. **Analyze** the completed code.
2. **Verify OOP principles** — update code as needed; confirm nothing broke.
3. **Scan for bugs and errors** — fix automatically.
4. **Automated code review:** Run `/code-review --fix` to auto-detect and apply reusability/simplification/efficiency cleanups. Review the diff before accepting.
5. **AI-generated code verification:**
   - **Dependency existence check:** Before adding any AI-suggested package, verify it exists on the actual registry (`npm view <pkg>`, `pip3 index versions <pkg>`, Maven Central search). Prevents **slopsquatting**.
   - **Extra SAST pass:** AI-generated code has a 15–25% vulnerability rate. Focus on injection flaws, hardcoded secrets, and insecure defaults.
   - **License verification:** Confirm AI-suggested dependencies use licenses compatible with the project type (§2).

---

## 9. Test → Debug → Fix Cycle

1. **Test** the code automatically as thoroughly as possible.
2. **Debug** any errors automatically.
3. **Fix** errors automatically.
4. **Repeat** until all tests pass and no errors remain.
5. **Learn** from prior fix attempts — do not repeat failed approaches.
6. **Autonomous completion (optional):** For well-defined acceptance criteria, use `/goal "<condition>"` to run the test-debug-fix cycle autonomously until the condition is met. Useful for "all tests pass," "build succeeds," or "coverage above 90%."
7. **Prompt the user** to test manually:
   - Provide a **terminal command** runnable from anywhere to start the program.
   - Provide a **quick-start summary** (prerequisites, setup steps, how to launch).

---

## 10. Summaries & Logs

- **Error/Bug Log** (`error_log.html`): Actively updated/appended log of all errors/bugs fixed, solutions, and reasoning.
- **Session Summary** (`session_summary.html`):
  - Section 1: Work completed in current session.
  - Section 2: Summary of prior work before this session.

---

## 11. Strength & Weakness Analysis

After code analysis:
1. Provide a list of **strengths and weaknesses** of the codebase.
2. **Prompt** the user on whether any weaknesses should be addressed in next steps.

---

## 12. Next Steps

At session/milestone end, create `next_steps/` folder with:
- Agent-formatted prompt (`.html`) explaining what to do next, with context files (as `.html`).
- `.txt` file with exact terminal command to execute the next-steps prompt.

---

## 13. Directory Index (`dir_index.html`)

`NAVIGATION.md` (§26) is the **primary navigation artifact** — agent-readable, intent-rich, and the right tool for "how is this project laid out." Prefer it.

`dir_index.html` is **optional and narrowly scoped.** A hand-maintained file listing drifts the moment a file is added — a stale index misleads more than no index, and agents read the live filesystem anyway. Create one only when **both** conditions hold:
- The folder is **browsed by humans** and is **asset-heavy or rarely-changing** (e.g. `screenshots/`, `logo1/`, `public/`, design-asset libraries) where a clickable browser index genuinely helps.
- It is **auto-generated**, never hand-maintained — wire a generator (e.g. a `generate-dir-index` script) into the pre-commit hook so it cannot go stale. If you can't automate it, don't create it.

Do **not** blanket every folder with `dir_index.html`, and do **not** hand-track deleted filenames — `git log --diff-filter=D` is the authoritative historical record.

---

## 14. CLAUDE.md Archiving

Maintain copies of CLAUDE.md in each project's main directory. First copy keeps original filename; subsequent copies appended with timestamp (e.g., `CLAUDE_2026-01-31_143022.md`).

---

## 15. GitHub & Annotated Code

When code is pushed to `main`:
1. Create folder with **exact codebase copy**, filenames appended `_annotated`.
2. Annotated files: **detailed comments** so a junior developer can learn, understand, fix, and expand. Must run without errors (comments only, no functional changes).
3. Commit and push as part of documentation section.

---

## 17. Environment & Tooling

### Version Pinning (Strict)
- **Node.js:** `.nvmrc` with exact version (e.g., `20.11.0`). `package-lock.json` committed.
- **Python:** `.python-version` with exact version (e.g., `3.12.3`). Lockfile committed.
- **Java:** Version specified in `pom.xml` `<maven.compiler.source>` / `<target>`.
- Reproducible builds required — exact versions locked in lockfiles, no floating ranges in production dependencies.

### Package Managers
| Language | Manager |
|---|---|
| **JavaScript / TypeScript** | `npm` |
| **Python** | `pip3` (with `requirements.txt` or `pyproject.toml`). **Note:** Use `pip3` and `python3` — `pip` and `python` are not available on this system. |
| **Java** | `Maven` |

### Linting & Formatting
- **JS/TS:** ESLint + Prettier. Auto-fix on save and pre-commit.
- **Python:** Black (formatter) + Ruff (linter). Auto-fix on save and pre-commit.
- **Java:** Checkstyle + google-java-format. Auto-fix on pre-commit.
- Config files (`eslint.config.js`, `.prettierrc`, `pyproject.toml`, `checkstyle.xml`) in project root. **Note:** ESLint flat config (`eslint.config.js` or `eslint.config.mjs`) is now required — the legacy `.eslintrc.*` format is deprecated.
- Formatting runs via pre-commit hooks — code that doesn't pass lint cannot be committed.

### Port Assignment
- Before assigning a port (dev servers, APIs, DBs, etc.), check if it's in use (`lsof -i :<port>` or equivalent).
- If occupied, auto-select an available port — don't prompt the user or fail silently.

### Claude Code Environment Variables (Auto-Exposed)
Auto-set by Claude Code and available in subprocesses, MCP servers, and Bash. Consider these when scoping environment access (§37):

| Variable | Scope | Purpose |
|---|---|---|
| `CLAUDE_CODE_SESSION_ID` | Subprocesses, MCP servers | Current session identifier |
| `CLAUDE_EFFORT` | Bash commands | Current effort level (low/medium/high/max) |
| `CLAUDE_PROJECT_DIR` | MCP stdio servers | Project working directory — **may leak path info** |
| `CLAUDECODE=1` | Subprocesses | Indicates code is running inside Claude Code |

---

## 18. Security

### Secret Management
- **Never commit** `.env` files, credentials, API keys, or tokens.
- Include `.env.example` with placeholder values and comments. Add `.env` patterns to `.gitignore` at project creation.

### Dependency Auditing
- Run security audits every build: `npm audit` / `pip3 audit` / `mvn dependency-check:check`.
- Critical/high vulnerabilities must be resolved before merging to `main`.

### OWASP Top-10 Scanning
- During code verification (§8), scan for: SQL injection, XSS, CSRF, insecure deserialization, broken auth, sensitive data exposure, security misconfiguration.
- **Agentic OWASP (ASI01–ASI10):** For AI agent projects, also scan against OWASP Top 10 for Agentic Applications (2026) — covers agent goal hijacking, tool misuse, identity/privilege abuse, agentic supply chain, unexpected code execution, memory/context poisoning, insecure inter-agent communication, cascading failures, human-agent trust exploitation, rogue agents.
- Fix automatically where possible; prompt user for architectural security decisions.

### Pre-Commit Secret Detection
- Configure `detect-secrets` or `gitleaks` in pre-commit hooks to prevent accidental credential commits.

### Build Artifact & Source Map Leakage Prevention
- **Never publish source map files** (`.js.map`, `.css.map`) in npm packages, PyPI distributions, or any public artifact — they expose the entire original source in readable form (ref: Claude Code 512K-line source leak via npm source maps, March 2026).
- **Pre-publish audit:** Always run `npm pack --dry-run` before `npm publish` to verify no `.map`, source, or debug files are included.
- **Use allowlist, not denylist:** Set `"files"` in `package.json` to explicitly list only compiled output (`dist/*.js`, `dist/*.d.ts`). This is safer than `.npmignore` because new file types are excluded by default.
- **Disable source maps for publish builds:** `"sourceMap": false` in `tsconfig.json` or a separate `tsconfig.publish.json`.
- **CI gate:** Fail the build if `npm pack --dry-run` output contains `.map` files.

---

## 19. Git Workflow

### Branching Strategy (GitHub Flow)
- **`main`** is always deployable. All work on **feature branches**.
- Branch naming: `feat/description`, `fix/description`, `docs/description`, `refactor/description`.
- All merges via **Pull Requests** with self-review. Delete branches after merge.

### Commit Message Convention (Conventional Commits)
Format: `<type>(<optional scope>): <description>`

| Type | Use |
|---|---|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `refactor:` | Code restructuring (no behavior change) |
| `test:` | Adding or updating tests |
| `chore:` | Build, tooling, dependency updates |
| `style:` | Formatting only (no logic change) |
| `perf:` | Performance improvement |

### Pre-Commit Hooks
Every project must configure:
1. Lint + auto-format (§17)
2. Secret detection (§18)
3. Type checking (where applicable)

Tools: `husky` + `lint-staged` (JS/TS), `pre-commit` framework (Python), Maven plugin (Java).

---

## 20. CI/CD

### GitHub Actions Pipeline
Every project: `.github/workflows/ci.yml` on every push and PR:
1. **Lint** — fail on errors
2. **Type-check** — TypeScript `tsc --noEmit`, Python `mypy`, etc.
3. **Test** — full suite with coverage reporting
4. **Security audit** — fail on critical/high
5. **Build** — compile/bundle

**Merge to `main` blocked** if any step fails.

### Deployment
Per-project decision. Document in Technical Design Document (§4). Options: Vercel, Railway, Fly.io, DigitalOcean, Render, AWS, self-hosted VPS. Must align with service abstraction (§6) — no hard lock-in.

---

## 21. Documentation Standards

### README (`README.html`)
Every project must have:
1. Project description — what it does, who it's for
2. Badges — build status, coverage, license
3. Setup/install instructions — prerequisites, environment setup, how to run
4. Architecture overview — high-level system diagram or description
5. API reference link
6. Contributing guidelines
7. License

### Auto-Generated API Documentation
- **Backend:** Swagger/OpenAPI at `/api-docs`.
- **Libraries:** TypeDoc (TS), Javadoc (Java), Sphinx (Python).
- Must stay in sync — regenerate as part of build step.

### Changelog (`CHANGELOG.html`)
Auto-generated from Conventional Commits using `conventional-changelog` or equivalent. Regenerated on each release/merge to `main`.

---

## 22. Testing Standards

### Coverage Target: 90% Line Coverage
- Minimum 90% required. Measured and reported in CI pipeline (§20).
- PRs below 90% should not merge without justification.
- **Before merging to `main`:** Prompt user whether to target **98%** before proceeding.

### Test Types
| Test Type | When Required | Scope |
|---|---|---|
| **Unit** | Every function/method with non-trivial logic | Single function/class in isolation |
| **Integration** | All API endpoints, database ops, service interactions | Multiple components together |
| **E2E** | Critical user flows (login, checkout, core features) | Full application stack |

### Test Frameworks
| Language | Unit/Integration | E2E / Browser | API Testing |
|---|---|---|---|
| **JS/TS** | Vitest | Playwright | Supertest |
| **Python** | pytest | Playwright | requests + pytest |
| **Java** | JUnit 5 (Maven Surefire) | Playwright | — |

---

## 23. Performance & Monitoring

### Performance Budgets (Web)
| Metric | Target |
|---|---|
| Largest Contentful Paint (LCP) | < 3 seconds |
| Interaction to Next Paint (INP) | < 200ms |
| Cumulative Layout Shift (CLS) | < 0.1 |
| JS bundle size (gzipped) | < 200KB |

Measure with Lighthouse CI in pipeline. Violations = warnings (not build-breaking) but must be fixed before production.

### Error Tracking (Sentry)
- All production apps integrate **Sentry** (free tier: 5K errors/month).
- Wrap behind `ErrorTracker` interface (§6 service abstraction — swappable).
- Source maps upload for JS/TS projects.

### Structured Logging
| Language | Library | Format |
|---|---|---|
| **JS/TS** | `pino` | JSON |
| **Python** | `structlog` | JSON |
| **Java** | SLF4J + Logback | JSON |

All logs: structured JSON with `timestamp`, `level`, `message`, `context`. No `console.log`/`print()` in production.

---

## 24. Accessibility

### WCAG 2.1 AA Compliance
All GUI and web projects must target **WCAG 2.1 Level AA**:
- **Color contrast:** 4.5:1 normal text, 3:1 large text.
- **Keyboard navigation:** All interactive elements reachable and operable via keyboard.
- **Screen reader support:** Semantic HTML, ARIA labels, meaningful alt text.
- **Focus management:** Visible focus indicators, logical tab order, focus trapping in modals.
- **Responsive:** Usable at 200% zoom without horizontal scrolling.

### Accessibility Testing
- **Automated:** `axe-core` in component tests. Lighthouse CI accessibility audits in pipeline.
- **Manual (key flows):** Screen reader (VoiceOver/NVDA), keyboard-only navigation, high-contrast mode.
- Violations tracked and resolved before production release.

---

## 25. Workspace Navigation Map

> The workspace navigation map (active projects, root-level tools, archive directories) lives in **project memory** at `_andy_ai_projects_2026/CLAUDE.md`, which loads automatically when working anywhere in that tree. Keeping it there — not here — avoids duplication and keeps the project list current. This global file holds only the reusable conventions (§0–§24, §26–§38) that apply to every project.

---

## 26. Project Navigation Map (`NAVIGATION.md`)

### Purpose
Every non-trivial project (5+ directories or 10+ files, or multiple modules/layers) must include `NAVIGATION.md` in its root directory.

### Required Sections

```
# [Project Name] — Navigation Map

> One-sentence project description.

## Architecture Overview
[2-3 sentences on architecture pattern and key design decisions.]

## Directory Map
| Directory | Layer/Role | Description |
|---|---|---|
| `src/models/` | Data Layer | Database models and schemas |
| `src/services/` | Business Logic | Core business operations |
| `src/api/` | Interface | REST/GraphQL endpoint handlers |
| `src/adapters/` | Integration | External service wrappers (swappable) |
| `tests/` | Testing | Unit, integration, and E2E tests |

## Key Entry Points
- **Application start:** `src/main.py`
- **API routes:** `src/api/routes.py`
- **Config:** `config/settings.yaml`

## Module Relationships
- API layer calls Services, never accesses Models directly
- Services call Adapters for external I/O
- Adapters implement interfaces defined in `src/ports/`

## External Dependencies
| Dependency | Purpose | Swappable? |
|---|---|---|
| PostgreSQL | Primary data store | Yes (any SQL via SQLAlchemy) |
| Sentry | Error tracking | Yes (ErrorTracker interface) |
```

### Maintenance
- Create on project init or when threshold exceeded. Update on directory/architecture changes.
- Focus on **20-30 most structurally significant directories** and entry points.
- Primary navigation artifact — preferred over `dir_index.html` (now optional/auto-generated, see §13). Complements the Technical Design Document (rationale).

---

## 27. Internet-Facing Security Verification

### Scope
Applies to **any code connected to the internet** — API endpoints, webhooks, web servers, WebSocket handlers, client apps, cloud functions, database connections exposed via network, OAuth/SSO integrations.

### Mandatory Checklist (Pre-Merge — Blocking)
Before internet-facing code merges to `main`, verify all items — document findings in `error_log.html` if automated scanning is unavailable.

#### Transport & Network
- [ ] **TLS 1.2+ enforced** — No plaintext HTTP in production
- [ ] **Certificate validation enabled** — No `rejectUnauthorized: false` or `verify=False` in production
- [ ] **CORS restrictive** — Explicit origins, no wildcard in production unless intentionally public and read-only
- [ ] **Rate limiting** on all public endpoints

#### Authentication & Authorization
- [ ] **Auth required** for all user data endpoints. No unprotected admin routes
- [ ] **Per-resource authorization** — Users access only own data. Broken access control (OWASP #1) tested
- [ ] **Secure tokens/sessions** — RS256/ES256 JWTs, short expiry, refresh rotation. Cookies: `Secure`, `HttpOnly`, `SameSite=Strict`
- [ ] **No secrets in client bundles** — API keys/tokens server-side only

#### Input & Output
- [ ] **All external input validated/sanitized** — Type, length, format, allowed values at system boundary
- [ ] **Parameterized queries only** — No string concatenation in SQL/NoSQL/ORM queries
- [ ] **Output encoding** — XSS prevention via framework auto-escaping
- [ ] **CSRF protection** on state-changing requests
- [ ] **File upload restrictions** — Magic byte validation (not just extension), size limits, storage outside web root

#### Dependencies & Configuration
- [ ] **Dependency audit clean** — No critical/high vulnerabilities
- [ ] **Security headers** — CSP, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, HSTS, `Referrer-Policy`
- [ ] **Generic client errors** — Stack traces/internal paths never in production responses
- [ ] **No default credentials** — No hardcoded passwords, default admin accounts, or dev backdoors

#### AI Agent Components
When internet-facing code includes an AI agent, also apply all of §36 (Lethal Trifecta, tool tiering, sandboxing, egress controls, memory/data isolation, inter-agent trust).

### Enforcement
- Checklist is **blocking** — must pass before merge, or a documented user exception.
- When in doubt whether code is internet-facing, **assume it is**.
- Reference current OWASP Top 10 and OWASP Top 10 for Agentic Applications (2026).

---

## 28. Data Hiding & Encapsulation

### Principle
All code must enforce **data hiding** and **encapsulation** — internal implementation details must never leak across module boundaries.

### Access Control
- **Default to private/protected.** All class fields use most restrictive access:

| Language | Default Field Visibility | Access Pattern |
|---|---|---|
| **TypeScript** | `private` or `#` (ES private) | Getters/setters or accessor methods |
| **Python** | `_` prefix (protected) or `__` (name-mangled) | `@property` decorators |
| **Java** | `private` | Getter/setter methods or records |

- **No public fields** unless pure DTO/record/dataclass explicitly marked as such.
- **Never expose mutable collections directly** — return defensive copies or immutable views (`Collections.unmodifiableList()`, `Object.freeze()`, `tuple()` from list).

### Encapsulation Boundaries
- **Classes own their data** — Business logic on a class's data belongs inside that class.
- **Law of Demeter** — No `a.getB().getC().doThing()` chains. Refactor so `a` exposes needed behavior directly.
- **Interfaces at module boundaries** — Cross-module communication through defined interfaces/public APIs only.
- **State changes through methods only** — External code never sets state directly; mutations go through methods that validate invariants.

### Verification (during §8)
Flag: public mutable fields, external property mutation bypassing setters, 3+ accessor chains, cross-module internal imports. Auto-fix straightforward cases (make field private, add accessor); flag architectural decisions to user.

---

## 29. Local File Management

Apply when: downloading, organizing, renaming files, creating folder structures, managing non-code files.

### File Organization
- **Descriptive naming** — Clear file/folder names
- **Date stamps:** `YYYY-MM-DD` format. **Versions:** `_v1`/`_v2`. **Backups:** `_backup_YYYYMMDD`. **Temp:** `tmp_`/`temp_` prefix.
- **Consistent structure** — Follow existing conventions, max 4-5 levels deep.

### Downloads
- Verify source. Check integrity (checksums when provided). Download to logical destination. Clean up temp files.

---

## 30. Research & Information Gathering

Apply when: researching technologies/tools/libraries, comparing options, investigating bugs.

- Define the question. Use appropriate sources. Cross-reference. Synthesize concisely.
- Prefer official docs. Check dates. Note uncertainties. Cite sources.
- **Comparisons:** Present as table (Option | Pros | Cons | Best For).
- **Technical research:** Summary, Details, Sources, Recommendations.

---

## 31. Simple Commands & Installations

Apply when: installing via package managers, running shell commands, environment setup.

- Use package managers (`brew`, `apt`, `npm`, `pip3`). Verify package names. Note versions. Check success.
- Explain before running. Use safe defaults. Check prerequisites. Report results.
- **Simple installs do NOT require GitHub repo creation.**

---

## 35. Proactive Context Compaction Management

### Purpose
Prevent loss of critical context during context-window compaction.

**Note (May 2026):** Claude Code now has built-in reactive compaction (overflow-seed optimization); the system handles *triggering*, so the agent no longer tracks context percentage manually. These rules supplement it with domain-specific preservation priorities — *what* to prioritize when compaction occurs.

### Pre-Compaction Trigger (10% Early)
When context utilization reaches **90% of the compaction threshold**, initiate proactive self-compaction immediately — do not wait for system auto-compact. This gives the agent control over *what* gets compacted.

### Compaction Depth (25% Less Aggressive)
Compact **25% less content** than system default would remove. Example: if system would remove 40%, proactive compaction removes only 30%.

### Priority Preservation Order

| Priority | Content Type | Action |
|---|---|---|
| **1 — Critical** | Original task prompt / user instructions | Never compact — preserve verbatim |
| **2 — Critical** | Original codebase context (files, architecture, schemas) | Never compact — preserve verbatim |
| **3 — High** | Pending/future task definitions | Preserve in full |
| **4 — High** | Current task state (in-progress work) | Preserve in full |
| **5 — Medium** | Completed task results (final outputs) | Summarize to key outcomes only |
| **6 — Low** | Completed task intermediate steps | Compact aggressively — retain only conclusion |
| **7 — Lowest** | Superseded/stale information | Remove entirely |

### Multi-Task Session Awareness
- Before compacting, **identify all remaining tasks** and their required context.
- **Never discard context an upcoming task depends on**, even if generated during a completed task.
- Retain artifacts (code, configs, schemas) that later tasks reference.

### Compaction Summary
After proactive compaction, note: (1) what was compacted and why, (2) what was preserved and why, (3) current context utilization.

---

## 36. AI Agent Security & Sandboxing

### When to Apply
Apply when building, deploying, or integrating AI agents that execute code, call tools, process untrusted input, access private data, or communicate with other agents — incl. LLM features, agentic workflows, RAG pipelines, and MCP systems.

**Reference:** `best_practices_reference/2026-03-13/report_ai_sandboxing_2026.html` for full technical details, tool comparisons, and incident case studies.

### Claude Code Auto Mode (Baseline Protection)
As of May 2026, Claude Code's **auto mode is default-on** (built-in exfiltration detection, better destructive-op classification). Baseline only — it protects the *agent's own actions*, not the *code the agent writes or deploys*, so it does **not** replace the sandboxing below.

### Security Guidance Plugin (Recommended)
Enable the **security guidance plugin** for production deployments — it monitors edits, diffs, and commits in real-time and flags dangerous patterns (hardcoded secrets, injection vectors, insecure defaults) before they reach production.

### Lethal Trifecta Check (Mandatory)
Before deploying any agentic system, evaluate whether it combines:
1. **Private data access** — reads emails, documents, databases, or user files
2. **Untrusted input exposure** — processes external content (emails, shared docs, web pages, user uploads)
3. **Exfiltration capability** — can make outbound requests (render images, call APIs, generate links)

If all three are present, the system is **inherently vulnerable**. Sandboxing must eliminate at least one condition — typically #3 via network egress controls or #1 via filesystem isolation.

### Isolation Technology Selection

| Risk Level | Minimum Isolation | Examples |
|---|---|---|
| **Untrusted code execution** | MicroVM (Firecracker, Kata) or gVisor | AI coding agents, code interpreters, user-submitted code |
| **Trusted agent with tool access** | Container with seccomp + AppArmor | Internal agents calling approved APIs |
| **Read-only analysis** | Standard container or WASM | Summarization, classification, embeddings |

- **Docker alone is insufficient** for untrusted AI code execution (shared kernel attack surface).
- **Python cannot be safely sandboxed at language level** — always use OS-level isolation.
- For GPU workloads, verify container escape mitigations (CVE-2025-23266).

### Tool Permission Tiering (OWASP)

| Tier | Permission | Approval | Examples |
|---|---|---|---|
| **Tier 1 — Read** | Read-only access | Auto-approved | File read, database SELECT, API GET |
| **Tier 2 — Write** | Create/modify resources | Review required | File write, database INSERT/UPDATE, API POST |
| **Tier 3 — Destructive** | Delete/irreversible | Fresh human approval per invocation | File delete, database DROP, account actions |

Never cache or persist approvals across executions. Show confidence and source attribution.

### Network Egress Controls
- **Default-deny outbound** — No network access by default.
- **Explicit allowlisting** — Only specific domains/IPs required for the task.
- **DNS filtering** — Block non-allowlisted domains to prevent DNS exfiltration.
- **Image URL exfiltration prevention** — Block/proxy agent-initiated image/resource loads (EchoLeak/GeminiJack vector).

### Ephemeral Execution
- Create and destroy sandboxes per session. No state accumulation between sessions.
- Persistent sandboxes: periodic recreation schedules (e.g., weekly resets).

### Monitoring & Audit Logging

| Metric | Threshold | Action |
|---|---|---|
| CPU usage per session | 80% of allocated | Throttle + alert |
| Memory per session | 512 MB default | Kill + alert |
| Execution duration | 5 min default | Kill + alert |
| Failed tool calls | 5 consecutive | Alert + review |
| Injection attempt detection | 1 | Alert + block + investigate |
| Sensitive data in output | Pattern-based (SSN, CC, keys) | Redact + log + alert |

- **Immutable audit trails** — Log all code execution, tool calls, API requests with write-once storage.
- **Automated redaction** — Credentials and API keys redacted before log persistence.
- **Anomaly detection** — Monitor for unexpected network connections, excessive API calls, unusual resource consumption.

### Multi-Agent Security
- **Cryptographic signatures** on all inter-agent messages.
- **Mutual TLS** required for inter-agent communication.
- **5-minute message freshness** window to prevent replay attacks.
- **Circuit breakers** to prevent cascading failures.
- **Per-agent least privilege** — each agent gets only permissions its role requires.

---

## 37. MCP Server Security

### When to Apply
Apply when installing, configuring, or using MCP servers — third-party, custom-built, or any tool-use infrastructure bridging AI models to local/remote systems.

### Vetting Policy (Before Installation)
1. **Source review** — Verify repo, known vulnerabilities, maintenance activity, contributor reputation.
2. **Permission audit** — Document: filesystem paths, network endpoints, system commands, env vars. Note `CLAUDE_PROJECT_DIR` is auto-passed to MCP stdio servers — exposing the project working directory path.
3. **Minimal scope** — Only specific directories/resources needed. Never blanket filesystem/network access.
4. **Pin version** — Specific commit or version. Never `latest` or floating references.
5. **`alwaysLoad` flag review** — If a server uses `alwaysLoad` (bypasses tool-search deferral, loads every session), document the justification. It increases attack surface by exposing all tools without lazy loading; approve only for servers genuinely needed every session.

### Isolation Requirements

| MCP Server Type | Minimum Isolation |
|---|---|
| **Third-party / untrusted** | Docker container with mount restrictions, no host network, read-only root FS |
| **First-party / internal** | Standard process isolation with restricted PATH and ENV |
| **Network-facing** | Docker + network namespace isolation, explicit port allowlisting |

- Network-accessing MCP servers must have **egress restrictions** per §36 network controls.
- MCP servers must **not** have write access to `CLAUDE.md`, `.mcp.json`, or agent behavior configs (prevents self-modification).

### Configuration Protection
- **`.mcp.json` integrity** — Security-sensitive config. Changes require explicit user review.
- **Environment variable scoping** — Only pass specific env vars needed. Never inherit full shell environment.
- **Secret injection** — Prefer short-lived tokens via credential brokers.

### Monitoring & Audit
- Log all MCP tool invocations: timestamp, tool name, parameters, caller identity, result status.
- Alert on: unexpected tool calls, outside normal patterns, privilege escalation attempts.
- Monthly review of MCP server access patterns for scope creep.

### Session Start Audit
When a new session begins in a project with MCP servers, verify:
1. Each MCP server is still needed and actively maintained.
2. No server has permissions beyond documented scope.
3. Version pins are current and not affected by known CVEs.

---

## 38. Dynamic Workflows & Background Agents

### Purpose
Leverage agent orchestration for parallel work across the workspace.

### Available Commands

| Command | Purpose | When to Use |
|---|---|---|
| `/goal "<condition>"` | Autonomous work until condition met | Test-debug-fix loops (§9), build targets, coverage goals |
| `/bg` | Spawn warm background session | Long-running builds, monitoring, parallel research |
| `claude agents` | Unified dashboard for all sessions | Track running, blocked, and completed agents |
| `/workflows` | View dynamic workflow runs | Monitor multi-agent orchestration |

### Usage Guidelines
- **`/goal`** — well-defined acceptance criteria ("all tests pass," "build succeeds," "lint clean," "coverage above 90%"). Avoid vague goals.
- **`/bg`** sessions stay warm — no timeout. Use for builds, test suites, or monitoring while working interactively.
- **Dynamic workflows** orchestrate work across multiple agents — useful for cross-project tasks (e.g., "update all Python projects to the new logging pattern").
- **Prefix shell commands with `!`** to run in the background and get results later.

### Security Considerations
- Background agents inherit the parent session's permission mode.
- `/goal` loops should be bounded — avoid open-ended goals that make unbounded changes.
- All background agent actions are logged in the same audit trail as interactive sessions.
