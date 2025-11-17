# Before (old code)

* We didn’t scan the paper text for counts.
* For each selected ontology category, we looked at the paper’s keywords already linked to that category (from the ontology) and asked: **how many of the query terms (incl. synonyms) appear in that set at least once?**
* Each term contributed **at most 1 point** per paper, even if it appeared many times in the title/abstract.
* Net effect: papers got credit for **distinct term presence** tied to the ontology, not for **how often** those terms are actually used in the paper text. This led to lots of ties and sometimes over-rewarded papers that touched many terms shallowly.

# After (new code — what we just changed)

* For each selected ontology category, we now **scan the title and abstract** and **count every occurrence** of each query term (incl. synonyms) with word boundaries.
* We sum those occurrence counts across the chosen categories: this is $M(p)$ exactly as in the paper.
* Each repeated mention adds more points, so dense on-topic papers rise; shallow mentions don’t get inflated credit.

# Practical impact

* **Fewer ties** in static and better separation: three papers with the same distinct terms but different densities will no longer all get the same score.
* Papers that truly focus on the query (more mentions in text) move up; papers that only brush the topic (one-off mentions) move down.

Essentially: we match the pre-print's implementation to make sure that our code is accurate. 
