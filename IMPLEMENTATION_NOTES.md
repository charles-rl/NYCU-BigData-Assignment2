# Data Notes

After inspecting 
```
Table                Total Rows   Unique PKs  
----------------------------------------------
IssueComments        2,595        2,595       
Issues               6,685        6,685       
PRReviewComments     1,475        1,475       
PRReviews            2,441        2,441       
PullRequests         10,075       10,075      
Pushes               42,116       42,116      
Repositories         11,475       11,475      
Users                5,338        5,338       
----------------------------------------------
Total Database Rows: 82,200
```
It seems that the number of database rows will fit inside my GPU and this is for October 1 to October 31 data.

Full 8 node heterogeneous graph is used because it matches with the schema graph and is **bidirectional** which is what I want.

---

## Temporal Design

The research finding: A 2025 paper showed that a simple 2 to 8-dimensional learnable linear projection beats complex trigonometric/sinusoidal encodings across almost all benchmarks.

So SOTA models use two distinct time channels which is the;
- Global anchor: How long before the cutoff did this specific event occur?
- User-level recency: How many days has it been since this user performed this activity?

The time needs to be normalized as well with mean and sd so that it doesn't mess up the network as usual in ML.

### TODO

**Not sure what this means yet and needs further learning**

Optimize decision thresholds per activity

- pr_review_comment only occurs in ~1.9% of user records.
- A standard 0.5 decision threshold will predict all zeros for rare activities. You must search for optimal per-label decision thresholds (e.g., 0.15 for reviews, 0.45 for pushes) to maximize F1.

---

## Feature Design

I noticed these text patterns when printing the database, despite them all getting masked with 'a'. There is still some patterns like;
`title            str  98.6%     140 ['aaaa(aaaaa): aaaaaaaaaa aaaaaa...` These aren't useful for NLPs or Language Embedding models but maybe we can extract some statistic from these text and get a few features that would help the model learn in the future. Maybe an ablation could be done to compare if it is actually useful.

```
================================================================================
TABLE: Issues (Total Rows: 6,685)
================================================================================
         Column          Dtype Null %  Unique                              Sample
       issue_id          int64   0.0%    6685                              [0, 1]
github_issue_id          Int64   0.0%    6685  [902409756721287, 748759002810787]
        repo_id          Int64   0.0%    1959                             [2, 10]
         number          Int64   0.0%    3383                        [5588, 1173]
      author_id          Int64   0.0%    1915                             [2, 10]
is_pull_request        boolean   0.0%       2                      [False, False]
pull_request_id          Int64  97.9%     140                            [86, 96]
          title            str   0.0%    6405 ['aaaa aaa aaaaa aaaaaaaa aaaaaa...
           body            str   2.3%    6490 ['### aaaaaaa aaaaaaaaaa\n\naaaa...
```

Doing a text pattern analysis, it is clear that there are patterns that I can extract and use for the model even if it is just small.
```
Raw Text Pattern                   What It Means                        Extracted Numeric Feature
──────────────────────────────────────────────────────────────────────────────────────────────────────────
PullRequests.head_ref              74% contain '/'                      • branch_depth (count of '/')
"aaa/000-aaaaaa-aaaa"              (e.g., feature/branch vs main)       • is_nested_branch (0 or 1)

PullRequests.base_ref              Median length = 4                    • is_main_branch (len <= 5)
"aaaa"                             (target is main/master)

Issues.title                       47% contain ':'                      • has_conventional_prefix (0/1)
"aa(aaaa): aaaa..."                28.6% contain '(' and ')'            • title_length

Issues.body                        Median length = 1,595 chars          • log_body_length
"## aaaaaaa ... ```"               82.5% contain '#' (markdown header)  • has_markdown_header (0/1)
                                   79.0% contain '`' (code blocks)      • has_code_block (0/1)

PRReviewComments.diff_hunk         98.6% contain '@', '+', '-'          • log_diff_length
"@@ -00,00 +00,00 @@"              Unified git diff snippet             • diff_line_count (count of '\n')

PRReviewComments.path              Median length = 41 chars             • file_depth (count of '/')
"aa/aaa/aaa/aaaa.aa"               96.2% contain '/'
```

If extracted, these are the only 6 features with enough non-null data to be viable:
1. **`Issues.body` → `log_issue_len`:** $\log(1 + \text{char\_length})$ (Distinguishes one-line issues from detailed reports).
2. **`Issues.body` → `has_markdown_structure`:** Binary flag for `#` or ` ``` ` (Distinguishes template bug reports from casual comments).
3. **`Issues.title` → `has_prefix`:** Binary flag for `:` or `(` (Captures conventional commit/issue prefixes like `feat:` or `fix:`).
4. **`PullRequests.head_ref` → `branch_depth`:** Count of `/` (Captures branch hierarchy, e.g. `feature/user/branch` vs `main`).
5. **`PRReviewComments.diff_hunk` → `log_diff_len`:** $\log(1 + \text{char\_length})$ (Captures diff inspection depth).
6. **`PRReviewComments.path` → `path_depth`:** Count of `/` (Captures directory depth of the modified file).

---

# Model Architecture part

## My decisions and why
- I pick full-graph batching here because my hardware can handle the massive data.
- I also picked a layernorm + mish to map the message in a latent space
- I choose HGTConv but I have to refer a paper to explain this decision. 
  - Relational Deep Learning (RDL): Fey et al., "Relational Deep Learning: Graph Representation Learning on Relational Databases" (arXiv:2312.04615, 2023). RelBench uses HGT as its primary heterogeneous graph baseline.
  - Original HGT Paper: Hu et al., "Heterogeneous Graph Transformer" (WWW 2020).
- Concatenating is good here to prevent the message passing diluting the user specific feature of time. We don't want this diluted with other features.
- Task-specific heads linear heads branching off the shared representation so that their losses won't affect each other from experience. Weighted BCE of course because of the rarity of some predictions.
