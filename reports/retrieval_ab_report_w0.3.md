# Retrieval A/B Report — local vs cross_area

Corpus: `support_tickets\support_tickets.csv` (89 tickets)  
`CROSS_AREA_WEIGHT` = **0.3**

## Headline

**LOCAL WINS — more disagreements favor local, avg coverage no worse.**

## Aggregate metrics

| Metric                                             |  Local | Cross-area |
|----------------------------------------------------|-------:|-----------:|
| Tickets with at least one retrieved chunk          |     89 |         89 |
| Tickets with zero chunks (no_docs path)            |      0 |          0 |
| Avg top-chunk term coverage                        | 0.3210 |     0.3188 |
| Avg top-chunk fused score                          | 0.8290 |     0.6510 |
| `area_in == 'none'` tickets grounded (cov >= 0.15) |   9/10 |       8/10 |

## Agreement

- Both modes pick the same top chunk: **82/89 (92.1%)**
- Modes disagree on top chunk: **7**
  - Cross-area has higher coverage: **1**
  - Local has higher coverage: **3**
  - Tie within tolerance (±0.02): **3**

## Top-chunk area distribution

| Area        | Local | Cross-area |
|-------------|------:|-----------:|
| devplatform |    39 |         40 |
| claude      |    27 |         27 |
| visa        |    23 |         22 |
| ?           |     0 |          0 |
| (empty)     |     0 |          0 |

## Per-ticket disagreements

Tickets where the two modes pick different top chunks. `cov` is the response-token-recall metric Stage-5 uses for weak-retrieval. `winner` = which mode found the higher-coverage chunk.

|           # | Subject                         | Company / area_in                                            |                                Winner                                 | Local top (area, cov)                                                                                                                              | Cross top (area, cov)                                                                                                                                                     |
|------------:|---------------------------------|--------------------------------------------------------------|:---------------------------------------------------------------------:|----------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|          23 | Delete unnecessary files        | None / none                                                  |                               **cross**                               | `data/claude/claude-api-and-console/api-faq/10366376-how-can-i-delete-my-claude-console-account.md` (claude, 0.250)                                | `data/devplatform/library/additional-resources/project-questions/1626384486-hidden-test-cases-for-front-end%2C-back-end-and-full-stack-questions.md` (devplatform, 0.375) |
|          24 | Tarjeta bloqueada               |                                                              |                                                                       |                                                                                                                                                    |                                                                                                                                                                           |
| Visa / visa | **tie**                         | `data/visa/support/consumer/travel-support.md` (visa, 0.087) | `data/visa/support/small-business/travelers-cheques.md` (visa, 0.087) |                                                                                                                                                    |                                                                                                                                                                           |
|          30 | Academic Research Request       | None / none                                                  |                               **local**                               | `data/devplatform/integrations/applicant-tracking-systems/greenhouse/1406188460-greenhouse---hackerrank-integration-guide.md` (devplatform, 0.361) | `data/claude/claude-for-nonprofits/12923235-using-the-candid-connector-in-claude.md` (claude, 0.222)                                                                      |
|          43 | Investment Advice Needed        | Visa / visa                                                  |                                **tie**                                | `data/visa/support/consumer/visa-rules.md` (visa, 0.207)                                                                                           | `data/visa/support/consumer/travel-support.md` (visa, 0.207)                                                                                                              |
|          63 | Visa card tier check            | Visa / visa                                                  |                               **local**                               | `data/visa/support/consumer/travel-support.md` (visa, 0.238)                                                                                       | `data/visa/support/consumer/visa-rules.md` (visa, 0.191)                                                                                                                  |
|          71 | No Subject                      | None / none                                                  |                                **tie**                                | `data/claude/claude-api-and-console/api-faq/8987200-can-i-use-the-claude-api-for-individual-use.md` (claude, 1.000)                                | `data/devplatform/screen/invite-candidates/9684438314-creating-an-email-template.md` (devplatform, 1.000)                                                                 |
|          82 | Security Alert — CVE-2026-41892 | None / none                                                  |                               **local**                               | `data/visa/support/small-business/data-security.md` (visa, 0.176)                                                                                  | `data/claude/claude-for-government/14503775-mcp-web-search.md` (claude, 0.118)                                                                                            |

