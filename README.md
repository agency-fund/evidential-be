# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/agency-fund/evidential-be/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                               |    Stmts |     Miss |   Cover |   Missing |
|----------------------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/xngin/apiserver/apikeys.py                                                     |       51 |        4 |     92% |     59-63 |
| src/xngin/apiserver/benchmarks/test\_draws\_perf.py                                |       95 |       66 |     31% |48, 54-95, 113-130, 145-161, 165, 173, 179-184, 196-219, 232-255, 268-286 |
| src/xngin/apiserver/certs/certs.py                                                 |       16 |        9 |     44% |19-23, 37-44 |
| src/xngin/apiserver/common\_field\_types.py                                        |       12 |        1 |     92% |        13 |
| src/xngin/apiserver/conftest.py                                                    |      245 |       25 |     90% |74, 93, 108, 110, 121, 164, 168, 170, 174, 407, 424-438, 453, 470, 473, 508 |
| src/xngin/apiserver/customlogging.py                                               |       66 |       13 |     80% |25-26, 48-68, 73-74, 102-107 |
| src/xngin/apiserver/database.py                                                    |       48 |        5 |     90% |30, 41, 58, 64, 71 |
| src/xngin/apiserver/dependencies.py                                                |       12 |        1 |     92% |        12 |
| src/xngin/apiserver/dns/safe\_resolve.py                                           |       69 |       15 |     78% |41-52, 88, 107, 112-113 |
| src/xngin/apiserver/dns/test\_safe\_resolve.py                                     |       35 |        1 |     97% |        49 |
| src/xngin/apiserver/dwh/dwh\_session.py                                            |      194 |       55 |     72% |71, 74, 137, 143, 154, 156-157, 168-223, 247-249, 264, 400-407, 411, 418-420, 439, 441-442, 453 |
| src/xngin/apiserver/dwh/dwh\_utils.py                                              |       17 |        3 |     82% |20, 27, 35 |
| src/xngin/apiserver/dwh/inspection\_types.py                                       |       55 |        5 |     91% |27, 45, 68, 79, 85 |
| src/xngin/apiserver/dwh/inspections.py                                             |       64 |        4 |     94% |67, 95, 100-103 |
| src/xngin/apiserver/dwh/participant\_metrics\_queries.py                           |      143 |        7 |     95% |73-79, 166, 263, 265, 289 |
| src/xngin/apiserver/dwh/queries.py                                                 |       81 |       27 |     67% |107-151, 162, 187, 211, 213, 233 |
| src/xngin/apiserver/dwh/query\_constructors.py                                     |       81 |        4 |     95% |75-76, 95-96 |
| src/xngin/apiserver/dwh/test\_dialect\_sql.py                                      |       74 |        6 |     92% |494, 507, 510-513 |
| src/xngin/apiserver/dwh/test\_queries.py                                           |       67 |        1 |     99% |       160 |
| src/xngin/apiserver/dwh/test\_query\_constructors.py                               |      160 |        2 |     99% |  444, 461 |
| src/xngin/apiserver/exceptionhandlers.py                                           |       68 |        9 |     87% |50, 62, 70-75, 89, 108 |
| src/xngin/apiserver/flags.py                                                       |       49 |        4 |     92% |73, 76, 94, 97 |
| src/xngin/apiserver/main.py                                                        |       38 |        6 |     84% |40, 48, 77-78, 95-97 |
| src/xngin/apiserver/openapi.py                                                     |       27 |        3 |     89% |100, 165, 167 |
| src/xngin/apiserver/pagination.py                                                  |      116 |       14 |     88% |47-48, 53-55, 61, 68, 98, 233, 235, 246-249 |
| src/xngin/apiserver/request\_encapsulation\_middleware.py                          |       69 |        3 |     96% |   113-115 |
| src/xngin/apiserver/routers/admin/admin\_api.py                                    |      733 |       47 |     94% |405, 417, 428, 432, 670, 1171, 1238-1244, 1430, 1467, 1588-1666, 1713, 1715-1718, 1758, 1879, 1942-1948, 1973, 1980, 2012, 2255 |
| src/xngin/apiserver/routers/admin/admin\_api\_converters.py                        |       62 |        8 |     87% |29, 73-74, 84, 110-111, 123-124 |
| src/xngin/apiserver/routers/admin/admin\_api\_types.py                             |      150 |        2 |     99% |    40, 42 |
| src/xngin/apiserver/routers/admin/test\_admin\_api.py                              |     1734 |        3 |     99% |2824, 2837-2838 |
| src/xngin/apiserver/routers/admin/test\_admin\_extra.py                            |      111 |        5 |     95% |98, 129-130, 158-159 |
| src/xngin/apiserver/routers/admin/test\_admin\_users\_api.py                       |      378 |        1 |     99% |        34 |
| src/xngin/apiserver/routers/admin\_integrations/admin\_integrations\_api.py        |      155 |        2 |     99% |  165, 404 |
| src/xngin/apiserver/routers/admin\_integrations/admin\_integrations\_api\_types.py |       15 |        1 |     93% |        26 |
| src/xngin/apiserver/routers/auth/auth\_api.py                                      |       65 |       37 |     43% |33-36, 65-76, 82-106, 111-150 |
| src/xngin/apiserver/routers/auth/auth\_dependencies.py                             |      139 |       32 |     77% |108, 121-127, 134-161, 258, 266, 279-281 |
| src/xngin/apiserver/routers/auth/test\_auth\_dependencies.py                       |      159 |        7 |     96% | 48, 55-61 |
| src/xngin/apiserver/routers/auth/token\_cryptor.py                                 |       43 |        4 |     91% |16-17, 53-54 |
| src/xngin/apiserver/routers/common\_api\_types.py                                  |      322 |       26 |     92% |145, 174, 176, 432, 434, 436, 493, 734-741, 760, 997, 1006, 1009-1010, 1020, 1022, 1032, 1034, 1353, 1515, 1517-1519 |
| src/xngin/apiserver/routers/common\_enums.py                                       |      201 |       39 |     81% |73, 75, 98-107, 112, 149-165, 187-188, 207-211, 249, 304-307, 316, 355-356, 360, 392 |
| src/xngin/apiserver/routers/experiments/dependencies.py                            |       60 |        3 |     95% |39, 60, 67 |
| src/xngin/apiserver/routers/experiments/experiments\_api.py                        |       96 |        4 |     96% |140-142, 364 |
| src/xngin/apiserver/routers/experiments/experiments\_common.py                     |      448 |       31 |     93% |252, 342-343, 366, 444, 455, 484-485, 496, 515, 606, 700-701, 723, 855, 859, 880-881, 884, 938-941, 960, 980, 1033-1034, 1050-1052, 1134 |
| src/xngin/apiserver/routers/experiments/experiments\_common\_csv.py                |       89 |        4 |     96% |43, 106, 240-241 |
| src/xngin/apiserver/routers/experiments/property\_filters.py                       |       96 |        8 |     92% |25, 28, 32, 95-96, 148, 160-161 |
| src/xngin/apiserver/routers/experiments/test\_experiments\_api.py                  |      454 |        3 |     99% |64, 178-179 |
| src/xngin/apiserver/routers/experiments/test\_experiments\_common.py               |     1181 |        9 |     99% |242-243, 253, 1573-1575, 2025-2026, 2479 |
| src/xngin/apiserver/routers/experiments/test\_property\_filters.py                 |       41 |        1 |     98% |        24 |
| src/xngin/apiserver/routers/healthchecks\_api.py                                   |       16 |        2 |     88% |     26-27 |
| src/xngin/apiserver/routers/integrations/integrations\_api.py                      |       51 |        1 |     98% |       173 |
| src/xngin/apiserver/routers/test\_assignment\_adapters.py                          |      235 |        1 |     99% |       103 |
| src/xngin/apiserver/settings.py                                                    |      143 |       23 |     84% |102, 122, 129, 135, 180-181, 241, 246, 252-253, 305-308, 327, 349, 360, 362, 372, 375, 389, 392, 417 |
| src/xngin/apiserver/snapshots/fake\_data.py                                        |      126 |       37 |     71% |73-81, 87, 90, 92, 97, 102, 107, 199-202, 280, 301-306, 322-349 |
| src/xngin/apiserver/snapshots/snapshotter.py                                       |       80 |        2 |     98% |  196, 207 |
| src/xngin/apiserver/snapshots/test\_snapshotter.py                                 |      255 |        8 |     97% |61-66, 566-567 |
| src/xngin/apiserver/sql/queries.py                                                 |       43 |       10 |     77% | 23, 61-70 |
| src/xngin/apiserver/sqla/tables.py                                                 |      338 |        4 |     99% |58, 223, 422, 426 |
| src/xngin/apiserver/storage/bootstrap.py                                           |       40 |        1 |     98% |        59 |
| src/xngin/apiserver/storage/storage\_format\_converters.py                         |      199 |       11 |     94% |49, 151, 156-157, 267, 305, 324, 358, 482, 543-544 |
| src/xngin/apiserver/testing/assertions.py                                          |        7 |        1 |     86% |         7 |
| src/xngin/cli/commands/create\_testing\_dwh.py                                     |      176 |      138 |     22% |31-33, 37-41, 52-83, 109, 120-130, 142-144, 149-156, 160-163, 167-171, 175-181, 188-212, 216-255, 259-264, 268-292, 377-397 |
| src/xngin/cli/common.py                                                            |       35 |       29 |     17% |     17-56 |
| src/xngin/cli/main.py                                                              |      206 |      136 |     34% |50-55, 65-69, 75-83, 123-130, 145-156, 169-180, 184-187, 229-277, 293-299, 310-311, 319-320, 327-390, 413-457, 461 |
| src/xngin/db\_extensions/custom\_functions.py                                      |       29 |        2 |     93% |    35, 55 |
| src/xngin/db\_extensions/test\_custom\_functions.py                                |       41 |        6 |     85% |     58-67 |
| src/xngin/events/common.py                                                         |       12 |        1 |     92% |        20 |
| src/xngin/events/experiment\_created.py                                            |       13 |        1 |     92% |        24 |
| src/xngin/ops/sentry.py                                                            |       13 |        6 |     54% |     18-40 |
| src/xngin/stats/assignment.py                                                      |       87 |        2 |     98% |  170, 262 |
| src/xngin/stats/balance.py                                                         |       74 |        2 |     97% |  110, 141 |
| src/xngin/stats/bandit\_analysis.py                                                |       73 |        4 |     95% |134, 136, 199-200 |
| src/xngin/stats/bandit\_sampling.py                                                |       86 |        7 |     92% |184, 219, 226, 254, 282, 284, 316 |
| src/xngin/stats/bandit\_weights\_to\_prior.py                                      |       50 |        2 |     96% |   75, 138 |
| src/xngin/stats/cluster\_icc.py                                                    |       40 |        2 |     95% |    33, 57 |
| src/xngin/stats/cluster\_power.py                                                  |      114 |        2 |     98% |  251, 254 |
| src/xngin/stats/individual\_power.py                                               |      103 |        5 |     95% |77, 80, 124-125, 200 |
| src/xngin/stats/power.py                                                           |       35 |        2 |     94% |  164, 169 |
| src/xngin/stats/stats\_errors.py                                                   |       25 |        3 |     88% |10, 37, 45 |
| src/xngin/tq/handlers.py                                                           |       85 |       11 |     87% |82-83, 129, 137-138, 149-158, 187-188 |
| src/xngin/tq/task\_queue.py                                                        |      100 |        2 |     98% |   241-242 |
| src/xngin/tq/tq\_test\_support.py                                                  |       49 |        5 |     90% |28-29, 47, 49, 71 |
| src/xngin/xsecrets/chafernet.py                                                    |       52 |        1 |     98% |        92 |
| src/xngin/xsecrets/gcp\_kms\_provider.py                                           |       70 |       28 |     60% |64-79, 86-87, 104-108, 111, 115-123, 127-134 |
| src/xngin/xsecrets/provider.py                                                     |       19 |        1 |     95% |        46 |
| src/xngin/xsecrets/secretservice.py                                                |       63 |        5 |     92% |37, 49-50, 108, 130 |
| src/xngin/xsecrets/test\_gcp\_kms\_provider.py                                     |      103 |       26 |     75% |40-42, 170-175, 182-189, 195-199, 206, 213-224 |
| src/xngin/xsecrets/test\_nacl\_provider.py                                         |       67 |        1 |     99% |        24 |
| **TOTAL**                                                                          | **15071** | **1101** | **93%** |           |

81 files skipped due to complete coverage.


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/agency-fund/evidential-be/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/agency-fund/evidential-be/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/agency-fund/evidential-be/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/agency-fund/evidential-be/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fagency-fund%2Fevidential-be%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/agency-fund/evidential-be/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.