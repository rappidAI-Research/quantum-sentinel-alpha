# Quantum Sentinel Alpha Threat Model

Status: initial Phase-0 threat model. This document defines product boundaries, not a claim of complete security.

## Assets to protect

- User repositories, diffs, source code, configuration and secrets.
- Integrity of findings, localization, verification state and patch output.
- Sentinel Garden provenance, rights metadata, manifests and split integrity.
- Private SentinelBench holdout cases and ground truth.
- Training/evaluation artifacts, model revisions, hashes and release provenance.

## Trust boundaries

1. **Repository input is untrusted.** Source files, comments, docs and agent instructions may contain adversarial text or prompt injection.
2. **Model output is untrusted until validated.** The model may hallucinate vulnerabilities, locations, CWE labels, evidence, tool requests or patches.
3. **Deterministic tools are outside the model.** Indexing, schema validation, static signals, test execution and enforcement cannot be delegated to model intent.
4. **Execution is a separate boundary.** Sentinel is not its own sandbox. Tests or tools must run through an isolated execution layer; Ghost may provide this later but is not required for model training.
5. **External data is untrusted until governed.** Public availability does not imply permission to train, redistribute or publish derived artifacts.

## Primary threats and required controls

| Threat | Failure mode | Initial control |
| --- | --- | --- |
| False positives | Plausible but nonexistent vulnerabilities overwhelm users | Hard negatives, precision-heavy evaluation, abstention, verifier state |
| Hallucinated evidence | Model invents source/sink paths or exploitability | Evidence fields, localization checks, re-analysis, explicit unverified state |
| Unsafe patch | Patch breaks behavior, removes features, or bypasses checks | Minimal diffs, isolated apply, functional/security tests, reject on failed verification |
| Prompt injection in repository | Repository text attempts to redefine policy or trigger tools | Treat repo content as untrusted context; tool policy stays deterministic outside model |
| Tool abuse | Model asks for unauthorized host/network actions | Narrow adapters and external authorization; optional Ghost runtime later |
| Secret leakage | Repo/tool context is copied into reports or model outputs unnecessarily | Minimize context, redact/avoid secrets, never treat model as secret boundary |
| Path/symlink escape | Indexer or patcher reads/writes outside repository | Canonical workspace boundary and path validation in Engine implementation |
| Dataset poisoning | Bad labels or adversarial samples degrade security behavior | Provenance registry, immutable revisions, verification artifacts, review gates |
| Rights/provenance failure | Training data cannot be legally used or redistributed | Per-source/repo license and terms review; nothing approved by default |
| Benchmark contamination | Training includes eval tasks/solutions | Repo-level and temporal splits, hashes/fuzzy matching, eval-only registry roles |
| Supply-chain drift | Foundation/tokenizer/dependencies change silently | Pin immutable revisions and hashes before paid training/release |
| Open-weight misuse | Modified weights remove defensive behavior | Do not claim tamper-proof safety; defensive training + transparent limitations + safe official Engine defaults |

## Explicit non-goals

Quantum Sentinel Alpha is not an autonomous remote penetration agent, internet-wide scanner, malware builder, credential-stealing assistant, ransomware assistant, binary/kernel exploitation specialist, or deterministic security enforcement layer.
