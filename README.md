# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/agency-fund/evidential-be/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                 |    Stmts |     Miss |   Cover |   Missing |
|--------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/xngin/apiserver/apikeys.py                                       |       51 |        4 |     92% |     59-63 |
| src/xngin/apiserver/certs/certs.py                                   |       16 |        9 |     44% |19-23, 37-44 |
| src/xngin/apiserver/common\_field\_types.py                          |       12 |        1 |     92% |        13 |
| src/xngin/apiserver/conftest.py                                      |      233 |       25 |     89% |65, 84, 99, 101, 112, 163, 165, 169, 382, 399-414, 424, 441, 444, 470 |
| src/xngin/apiserver/customlogging.py                                 |       66 |       13 |     80% |25-26, 48-68, 73-74, 102-107 |
| src/xngin/apiserver/database.py                                      |       48 |        5 |     90% |30, 41, 58, 64, 71 |
| src/xngin/apiserver/dependencies.py                                  |       12 |        4 |     67% | 12, 23-25 |
| src/xngin/apiserver/dns/safe\_resolve.py                             |       50 |       12 |     76% |32-36, 40-41, 54, 76, 79, 84-85 |
| src/xngin/apiserver/dwh/dwh\_session.py                              |      189 |       65 |     66% |75, 78, 141, 147, 158, 160-161, 172-229, 249-251, 266, 332-337, 341, 348-350, 369, 371-372, 383, 418-425, 433 |
| src/xngin/apiserver/dwh/dwh\_test\_support.py                        |       84 |        1 |     99% |       175 |
| src/xngin/apiserver/dwh/inspection\_types.py                         |       55 |        5 |     91% |27, 45, 68, 79, 85 |
| src/xngin/apiserver/dwh/inspections.py                               |       64 |        4 |     94% |67, 95, 100-103 |
| src/xngin/apiserver/dwh/participant\_metrics\_queries.py             |      143 |        7 |     95% |73-79, 166, 263, 265, 289 |
| src/xngin/apiserver/dwh/queries.py                                   |       79 |       44 |     44% |108-152, 162-192, 213-234 |
| src/xngin/apiserver/dwh/query\_constructors.py                       |       81 |        6 |     93% |37-39, 104-105, 124-125 |
| src/xngin/apiserver/dwh/test\_dialect\_sql.py                        |       74 |        6 |     92% |544, 557, 560-563 |
| src/xngin/apiserver/dwh/test\_query\_constructors.py                 |      135 |        2 |     99% |  453, 470 |
| src/xngin/apiserver/exceptionhandlers.py                             |       68 |        9 |     87% |50, 62, 70-75, 89, 108 |
| src/xngin/apiserver/flags.py                                         |       51 |        4 |     92% |76, 79, 97, 100 |
| src/xngin/apiserver/main.py                                          |       38 |        6 |     84% |40, 48, 76-77, 94-96 |
| src/xngin/apiserver/openapi.py                                       |       22 |       13 |     41% |     22-86 |
| src/xngin/apiserver/pagination.py                                    |      108 |       13 |     88% |47-48, 53-55, 61, 68, 220, 222, 233-236 |
| src/xngin/apiserver/request\_encapsulation\_middleware.py            |       69 |        3 |     96% |   113-115 |
| src/xngin/apiserver/routers/admin/admin\_api.py                      |      629 |       54 |     91% |250-269, 312, 411, 423, 434, 438, 447, 636, 861, 865, 924-930, 1113, 1150, 1267-1345, 1392, 1394-1397, 1437, 1559, 1564, 1624, 1637-1638, 1663, 1941, 1947 |
| src/xngin/apiserver/routers/admin/admin\_api\_converters.py          |       62 |        9 |     85% |29, 73-74, 84, 110-111, 118, 123-124 |
| src/xngin/apiserver/routers/admin/admin\_api\_types.py               |      125 |        2 |     98% |    39, 41 |
| src/xngin/apiserver/routers/admin/test\_admin.py                     |     1493 |        1 |     99% |       384 |
| src/xngin/apiserver/routers/auth/auth\_api.py                        |       63 |       35 |     44% |33-36, 65-76, 82-106, 111-148 |
| src/xngin/apiserver/routers/auth/auth\_dependencies.py               |      137 |       32 |     77% |99, 112-118, 125-152, 249, 257, 270-272 |
| src/xngin/apiserver/routers/auth/test\_auth\_dependencies.py         |      159 |        7 |     96% | 48, 55-61 |
| src/xngin/apiserver/routers/auth/token\_cryptor.py                   |       43 |        4 |     91% |16-17, 53-54 |
| src/xngin/apiserver/routers/common\_api\_types.py                    |      319 |       27 |     92% |122, 148, 150, 426, 428, 430, 487, 768-775, 813, 1030, 1039, 1042-1043, 1053, 1055, 1065, 1067, 1141, 1302, 1474, 1476-1478 |
| src/xngin/apiserver/routers/common\_enums.py                         |      201 |       40 |     80% |73, 75, 98-107, 112, 149-165, 187-188, 207-211, 252, 287, 303-306, 315, 354-355, 359, 391 |
| src/xngin/apiserver/routers/experiments/dependencies.py              |       61 |        3 |     95% |39, 60, 67 |
| src/xngin/apiserver/routers/experiments/experiments\_common.py       |      418 |       31 |     93% |207, 232, 273-274, 301, 383, 397, 424, 436, 545, 604-605, 679-680, 701, 805, 809, 813, 870-871, 874, 913, 931, 939, 954, 1007-1008, 1024-1026, 1099 |
| src/xngin/apiserver/routers/experiments/experiments\_common\_csv.py  |       39 |        2 |     95% |     81-82 |
| src/xngin/apiserver/routers/experiments/property\_filters.py         |       96 |        8 |     92% |25, 28, 32, 95-96, 148, 160-161 |
| src/xngin/apiserver/routers/experiments/test\_experiments\_api.py    |      304 |        3 |     99% |59, 162-163 |
| src/xngin/apiserver/routers/experiments/test\_experiments\_common.py |     1047 |        8 |     99% |188-189, 1221-1223, 1724-1725, 2186 |
| src/xngin/apiserver/routers/experiments/test\_property\_filters.py   |       41 |        1 |     98% |        24 |
| src/xngin/apiserver/routers/healthchecks\_api.py                     |       16 |        2 |     88% |     26-27 |
| src/xngin/apiserver/routers/test\_assignment\_adapters.py            |      208 |        1 |     99% |       103 |
| src/xngin/apiserver/settings.py                                      |      143 |       24 |     83% |101, 121, 128, 134, 179-180, 240, 245, 251-252, 303-307, 326, 348, 359, 361, 371, 374, 388, 391, 416 |
| src/xngin/apiserver/snapshots/fake\_data.py                          |      126 |       37 |     71% |73-81, 87, 90, 92, 97, 102, 107, 199-202, 280, 301-306, 322-349 |
| src/xngin/apiserver/snapshots/snapshotter.py                         |       76 |       23 |     70% |101-102, 114-139, 167-190, 203 |
| src/xngin/apiserver/sql/queries.py                                   |       38 |       10 |     74% | 22, 60-69 |
| src/xngin/apiserver/sqla/tables.py                                   |      308 |        4 |     99% |57, 155, 354, 358 |
| src/xngin/apiserver/storage/bootstrap.py                             |       44 |        1 |     98% |        67 |
| src/xngin/apiserver/storage/storage\_format\_converters.py           |      169 |        8 |     95% |57-59, 274, 310, 452, 509-510 |
| src/xngin/apiserver/testing/admin\_api\_client.py                    |      186 |       14 |     92% |158-161, 206, 223-230, 255-259, 315, 1655, 2601, 3819 |
| src/xngin/apiserver/testing/assertions.py                            |        7 |        1 |     86% |         7 |
| src/xngin/apiserver/testing/experiments\_api\_client.py              |       96 |       11 |     89% |164, 181-188, 202-203, 213-217, 224, 535 |
| src/xngin/apiserver/testing/pg\_helpers.py                           |       13 |        2 |     85% |     16-17 |
| src/xngin/db\_extensions/custom\_functions.py                        |       39 |        3 |     92% |47, 66, 79 |
| src/xngin/db\_extensions/test\_custom\_functions.py                  |       46 |       12 |     74% |59-68, 73-82 |
| src/xngin/events/common.py                                           |        7 |        2 |     71% |    14, 21 |
| src/xngin/events/experiment\_created.py                              |       13 |        1 |     92% |        24 |
| src/xngin/events/webhook\_sent.py                                    |       12 |        5 |     58% |     18-23 |
| src/xngin/ops/sentry.py                                              |       13 |        6 |     54% |     18-40 |
| src/xngin/stats/assignment.py                                        |       59 |        1 |     98% |       128 |
| src/xngin/stats/balance.py                                           |       67 |        2 |     97% |  110, 138 |
| src/xngin/stats/bandit\_analysis.py                                  |       75 |        5 |     93% |135, 137, 139, 202-203 |
| src/xngin/stats/bandit\_sampling.py                                  |       73 |        7 |     90% |162, 186, 191, 214, 238, 240, 272 |
| src/xngin/stats/bandit\_weights\_to\_prior.py                        |       50 |        2 |     96% |   75, 138 |
| src/xngin/stats/cluster\_icc.py                                      |       40 |        2 |     95% |    33, 57 |
| src/xngin/stats/power.py                                             |      118 |        5 |     96% |71, 74, 118-119, 195 |
| src/xngin/xsecrets/chafernet.py                                      |       52 |        1 |     98% |        92 |
| src/xngin/xsecrets/gcp\_kms\_provider.py                             |       70 |       28 |     60% |64-79, 86-87, 104-108, 111, 115-123, 127-134 |
| src/xngin/xsecrets/provider.py                                       |       19 |        1 |     95% |        46 |
| src/xngin/xsecrets/secretservice.py                                  |       63 |        5 |     92% |37, 49-50, 108, 130 |
| src/xngin/xsecrets/test\_gcp\_kms\_provider.py                       |      103 |       26 |     75% |40-42, 170-175, 182-189, 195-199, 206, 213-224 |
| src/xngin/xsecrets/test\_nacl\_provider.py                           |       67 |        1 |     99% |        24 |
| **TOTAL**                                                            | **11602** |  **781** | **93%** |           |

66 files skipped due to complete coverage.


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