Here’s what each constant means, why those specific values, and the effect they have:

* **a = 1.0 — ontology evidence (cat\_hit) weight**
  This is the “unit scale.” Each matched ontology keyword in the matched category is worth \~1 point. Setting **a=1.0** makes ontology overlap the primary driver and anchors the score scale so other terms are chosen relative to it. If you care more about strict ontology agreement, raise **a**.

* **b = 0.5 — text support (text\_hit) weight**
  Title/abstract term frequency backs up the ontology match, but we keep it **half** as strong (0.5) so raw frequency doesn’t swamp ontology structure (e.g., long abstracts or repeated terms). If you notice noisy boosts from verbose papers, lower **b**; if good papers are under-rewarded due to rich textual evidence, raise **b**.

* **γ = 0.3 — layer coverage bonus**
  Rewards touching more of the query-relevant categories **within a layer**. It appears inside a multiplier

   ![equation](https://latex.codecogs.com/svg.latex?%5Ccolor%7Bwhite%7D%5Cbigl%281%20%2B%20%5Cgamma%20%5Ccdot%20%5Ctext%7Blay%5C_cov%7D%20%2B%20%5Cdelta%20%5Ccdot%20%5Ctext%7Bcons%5C_norm%7D%5Cbigr%29)

  With **γ=0.3**, full coverage can add up to +30% to that layer’s score—enough to matter, not enough to overwhelm base evidence. If you want broader coverage to matter more, increase **γ**.

* **δ = 0.3 — within-layer consistency bonus**
  Encourages **balanced** evidence across categories in a layer using a normalized harmonic mean. Same magnitude as γ so coverage and balance are co-equal nudges. If single-spike categories are over-winning, raise **δ**; if you want to allow “one strong category is enough,” lower **δ**.

* **κ = 0.5 — cross-layer coherence bonus**
  Adds **0.5** per distinct layer-pair where the paper shows evidence. For example, with 3 active layers, the max pair count is 3, so at most **+1.5**—comparable to a couple of ontology hits, not dominant. This favors papers connecting multiple conceptual layers (e.g., composition + processing) without drowning out per-layer scores. If cross-layer integration is critical, raise **κ**.

### Why these exact numbers?

* They enforce **ontology-first** ranking (a > b),
* Give **moderate**, bounded boosts for coverage/consistency (γ, δ ≤ 0.3 → multiplier in \[1, 1.6]),
* And a **measured** cross-layer bonus (κ=0.5) that can help break ties but won’t reorder everything on its own.

### Quick tuning guide

* Too many verbose-but-off papers rising? ↓ **b** (e.g., 0.3).
* Great multi-category papers not surfacing? ↑ **γ** (coverage) or ↑ **δ** (balance).
* You want multi-layer integration to matter more? ↑ **κ** (e.g., 0.7–1.0).
* Want stricter ontology match dominance? ↑ **a** or ↓ **b**.

Typical sensible ranges: **a ∈ \[0.8, 1.5]**, **b ∈ \[0.2, 0.8]**, **γ, δ ∈ \[0.1, 0.5]**, **κ ∈ \[0.2, 1.0]**.