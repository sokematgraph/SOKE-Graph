## 🔄 Key Changes Made

### 1. **HRM configuration extras**

Added new attributes in `__init__` for the paper-style HRM:

```python
self.hrm_N = getattr(self, "hrm_N", 3)          # high-level cycles
self.hrm_T = getattr(self, "hrm_T", 4)          # low-level steps per cycle
self.hrm_use_act = True                         # enable ACT/Q-head
self.hrm_Mmax = 4                               # max segments for deep supervision/ACT
self.hrm_eps_minseg = 0.25                      # epsilon for random longer min-segment
```

➡️ These parameters implement the **two-level recurrent structure** (N cycles of T steps) and optional **Adaptive Computation Time (ACT)**.

---

### 2. **Ranker uses fallback HRS vs Neural HRM**

In `rank_papers()` we updated the HRM logic:

* **If** PyTorch + model weights/labels are available → run **neural HRM** (`_score_with_neural_hrm`).
* **Else** → log a message and fall back to the **ontology-based hierarchical reasoning scorer (HRS)**.

Before: the fallback was just a **linear combo** `α*static + β*pair_count`.
Now: it’s a proper **hierarchical reasoning scorer** (`_score_with_hrs`).

---

### 3. **New training loop** (`_train_neural_hrm`)

* Implements **deep supervision**: multiple forward passes over M segments with detaching in between.
* Supports **pairwise** (`pid_pos,pid_neg`) or **pointwise** (`pid,label`) label files.
* Adds **optional ACT loss** via a Q-head (decides halt/continue).
* Saves trained weights to `hrm.pt`.

Before: HRM training was a simple BCE loop.
Now: it matches the paper’s **1-step gradient approximation** idea.

---

### 4. **New HRM model class** (`_NeuralHRM`)

Replaced the old simple GRU scorer with a **two-level recurrent architecture**:

* **Low-level GRU (`l_rnn`)** runs T steps per cycle.
* **High-level GRU (`h_rnn`)** updates once per cycle from the final L-state.
* **LayerNorm** stabilizes both.
* **Linear scorer** outputs relevance.
* **Q-head** outputs `[halt, continue]` probabilities if ACT is enabled.

This is the biggest change: it makes the HRM actually resemble the **paper’s architecture**.

---

### 5. **Integration in `_score_with_neural_hrm`**

Changed the forward call to use the new HRM interface:

```python
scores = self._hrm_model.forward(x, N=self.hrm_N, T=self.hrm_T)
```

instead of the old “average GRU states over K steps” approach.

---

### 6. **Fallback Hierarchical Reasoning Scorer (`_score_with_hrs`)**

If no neural HRM, we now run a structured scorer that:

* Combines **ontology overlap** and **text hits** for each category.
* Aggregates per layer with **coverage** and **consistency bonuses**.
* Adds a **cross-layer coherence bonus**.

This ensures the fallback itself is still ontology-aware and “hierarchical,” not just a linear combo.

---

## ✅ Summary

In short, the changes we made turned HRM from a **basic GRU + linear head (stub)** into a **paper-style Hierarchical Recurrent Model** with:

* Two recurrent modules (low-level and high-level).
* Multi-cycle updates (N × T).
* Deep supervision + detach training.
* Optional ACT halting.
* Ontology-aware fallback HRS if no weights/labels exist.

And — we kept the **pipeline and CSV outputs the same**, so nothing downstream breaks.

