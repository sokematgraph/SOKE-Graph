A

## 📊 Setup

Ontology (2 layers × 2 categories each):

```
Layer A:
  - Cat1: {“nickel”, “cobalt”}
  - Cat2: {“cheap”, “abundant”}

Layer B:
  - Cat3: {“durable”, “stable”}
  - Cat4: {“fast”, “efficient”}
```

User query → `"cheap durable"`

So the query keywords hit:

* Layer A / Cat2 → “cheap”
* Layer B / Cat3 → “durable”

---

## 📄 Candidate Papers

* **P1**: Title contains “cheap nickel catalyst is durable”
* **P2**: Title contains “fast nickel process”

---

## 🔢 Scoring with `_score_with_hrs`

### Step 1. Category-level scoring

Formula:

```
CatScore = a * ontology_hits + b * text_hits
```

(let’s assume a=1.0, b=0.5 for simplicity)

* **P1**

  * Layer A / Cat2: matches “cheap” → ontology\_hits=1, text\_hits=1 → score = 1 + 0.5\*1 = 1.5
  * Layer B / Cat3: matches “durable” → ontology\_hits=1, text\_hits=1 → score = 1.5

* **P2**

  * Layer A / Cat2: no match → score=0
  * Layer B / Cat3: no match → score=0

---

### Step 2. Layer aggregation

Each layer’s score = sum(cat scores) × (coverage + consistency bonuses).
Let’s simplify: here each layer has just 1 relevant category hit, so bonuses ≈ 1.

* **P1**:

  * Layer A = 1.5
  * Layer B = 1.5

* **P2**:

  * Layer A = 0
  * Layer B = 0

---

### Step 3. Cross-layer coherence

Bonus = κ × (#layer\_pairs). κ=0.5 here.

* **P1**: appears in both Layer A and B → 1 pair → bonus=0.5
* **P2**: appears in no layers → bonus=0

---

### Step 4. Final scores

* **P1**: 1.5 + 1.5 + 0.5 = **3.5**
* **P2**: 0 → **0**

---

## 🪙 Old linear fallback (α*static + β*pair\_count)

Say static counts = how many keywords matched, pair\_count = overlaps across categories.

* **P1**:

  * static\_score = 2 (“cheap”, “durable”)
  * pair\_count = 0 (only 1 category per layer, no overlap)
  * score = α*2 + β*0 = 2.0

* **P2**:

  * static\_score = 0
  * pair\_count = 0
  * score = 0

---

## ✅ Comparison

| Paper | HRS Score (new) | Linear Fallback (old) |
| ----- | --------------- | --------------------- |
| P1    | 3.5             | 2.0                   |
| P2    | 0.0             | 0.0                   |

👉 Notice how P1 is **rewarded more strongly** in HRS because:

* It covers **multiple ontology layers** (A + B).
* It gets an extra **cross-layer bonus**.

The old linear fallback only saw “2 keyword hits” and didn’t care about structure.

---

