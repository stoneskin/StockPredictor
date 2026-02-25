# 📦 Archive Documentation

Deprecated, obsolete, or reference-only documentation. These files are preserved for historical context but **should not be used in new development**.

---

## Why Archive Exists

As the Stock Predictor project evolved:
- **V1** (Regression) was replaced by **V2** (Classification) for better results
- Early documentation becoming outdated
- Multiple revisions created overlapping/duplicate docs
- **Redesign documents** showing iteration history

Archive preserves this history for learning and reference.

---

## Contents by Category

### 1. Code Review & Analysis
**Files**: `codeReview-mm25.md`  
**Status**: Outdated  
**Why**: Code review from earlier iteration - structure has changed  
**Use Case**: Understand old feedback (not applicable now)

---

### 2. Early Redesign Documents  
**Files**: `REDESIGN.md`, `REDESIGN_V2.md`  
**Status**: Historical  
**Why**: Show evolution from REDESIGN → REDESIGN_V2 → final structure  
**Use Case**: Understand design decisions and why things changed

---

### 3. Chinese Documentation
**Files**: `完整实施方案.md`, `策略分析与ML应用建议.md`  
**Status**: Reference only  
**Why**: Original Chinese implementation plans and strategy analysis  
**Use Case**: Reference if working with original team/stakeholders

---

### 4. Duplicate/Superseded  
**Files**: `Stock Prediction with SageMaker.md`, `MIGRATION_GUIDE.md`, `IMPLEMENTATION_GUIDE.md`  
**Status**: Superseded by versioned docs  
**Why**: Now covered in `docs/v1/` and `docs/v2/` respectively  
**Use Case**: Reference old implementation approach

---

### 5. Technical Reference  
**Files**: `Pine Script - Vegas Channel + Hull STRG.md`, `Pine Script -MACD-RSI.md`  
**Status**: Educational  
**Why**: Pine Script strategies (not used in Python prediction system)  
**Use Case**: Reference technical indicator logic

---

### 6. Initial READMEs
**Files**: `README2.md`  
**Status**: Obsolete  
**Why**: Replaced by current [../../README.md](../../README.md)  
**Use Case**: Historical reference only

---

## File Inventory

| File | Type | Status | Reason |
|------|------|--------|--------|
| codeReview-mm25.md | Review | ⚠️ Outdated | Old code feedback |
| REDESIGN.md | Design | 📖 Reference | Earlier redesign |
| REDESIGN_V2.md | Design | 📖 Reference | V2 redesign |
| 完整实施方案.md | Strategy | 📖 Reference | Original Chinese plan |
| 策略分析与ML应用建议.md | Analysis | 📖 Reference | Strategy analysis |
| Stock Prediction with SageMaker.md | Deployment | 📖 Reference | Old SageMaker docs |
| MIGRATION_GUIDE.md | Guide | 📖 Reference | Now in v1/ & v2/ |
| IMPLEMENTATION_GUIDE.md | Guide | 📖 Reference | Superseded |
| Pine Script - Vegas Channel + Hull STRG.md | Technical | 📖 Reference | Pine Script only |
| Pine Script -MACD-RSI.md | Technical | 📖 Reference | Pine Script only |
| README2.md | README | ❌ Obsolete | Use main README |

---

## When to Reference Archive

✅ **DO read archive if**:
- Understanding historical design decisions
- Learning how the project evolved
- Working with original Chinese documentation
- Researching Pine Script technical indicators
- Studying old code review feedback

❌ **DON'T use archive if**:
- Building new features (use `docs/v2/`)
- Deploying system (use `docs/v2/`)
- Learning stock prediction (use `docs/v2/`)
- Troubleshooting current issues (use main README)
- New to project (use `GETTING_STARTED.md`)

---

## Current Documentation Structure

### Main Docs (Active)
See [../../](../../):
- **README.md** - Project overview
- **GETTING_STARTED.md** - Quick start
- **ARCHITECTURE.md** - System design
- **API_REFERENCE.md** - API documentation
- **V2_CLASSIFICATION.md** - Classification approach
- **TROUBLESHOOTING.md** - Problem solving

### Versioned Docs
- **[../v1/README.md](../v1/README.md)** - V1 historical context
- **[../v2/README.md](../v2/README.md)** - V2 current system

### Legacy Code
- **[../../src/v1/](../../src/v1/)** - V1 implementation (reference only)
- **[../../src/](../../src/)** - V2 implementation (active)

---

## Archive Structure

```
docs/archive/
├── codeReview-mm25.md                           # Old code review
├── REDESIGN.md                                  # Early redesign
├── REDESIGN_V2.md                               # V2 redesign
├── 完整实施方案.md                              # Chinese: Full implementation plan
├── 策略分析与ML应用建议.md                      # Chinese: Strategy analysis
├── Stock Prediction with SageMaker.md           # Old SageMaker guide
├── MIGRATION_GUIDE.md                           # Superseded migration guide
├── IMPLEMENTATION_GUIDE.md                      # Superseded implementation
├── Pine Script - Vegas Channel + Hull STRG.md  # Pine Script reference
├── Pine Script -MACD-RSI.md                     # Pine Script reference
└── README2.md                                   # Old README
```

---

## Moving Forward

### For New Contributors
1. Start with [../../GETTING_STARTED.md](../../GETTING_STARTED.md)
2. Read [../../ARCHITECTURE.md](../../ARCHITECTURE.md)
3. Refer to [../v2/README.md](../v2/README.md) for current system
4. **DO NOT** use archive files unless specifically researching history

### For Historians & Researchers
1. Explore [../v1/README.md](../v1/README.md) to understand V1
2. Review `REDESIGN.md` and `REDESIGN_V2.md` for design evolution
3. Check `完整实施方案.md` for original strategy
4. Reference Pine Script docs for technical indicators

### For Chinese Speakers
1. See `完整实施方案.md` - Full implementation plan
2. See `策略分析与ML应用建议.md` - Strategy & ML recommendations

---

## See Also

- **Current Documentation**: [../../](../../)
- **V1 Historical Context**: [../v1/README.md](../v1/README.md)
- **V2 Current System**: [../v2/README.md](../v2/README.md)
- **Tests**: [../../tests/README.md](../../tests/README.md)
- **Getting Started**: [../../GETTING_STARTED.md](../../GETTING_STARTED.md)

---

**Remember**: This archive is for reference only. Use [../../](../../) for current guidance. 📚