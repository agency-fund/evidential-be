# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/agency-fund/evidential-be/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                 |    Stmts |     Miss |   Cover |   Missing |
|--------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/xngin/apiserver/apikeys.py                                       |       51 |        4 |     92% |     59-63 |
| src/xngin/apiserver/certs/certs.py                                   |       16 |        9 |     44% |19-23, 37-44 |
| src/xngin/apiserver/common\_field\_types.py                          |       12 |        1 |     92% |        13 |
| src/xngin/apiserver/conftest.py                                      |      223 |       12 |     95% |67, 86, 101, 103, 114, 165, 167, 171, 383, 400, 403, 429 |
| src/xngin/apiserver/customlogging.py                                 |       66 |       13 |     80% |25-26, 48-68, 73-74, 102-107 |
| src/xngin/apiserver/database.py                                      |       46 |        5 |     89% |27, 38, 55, 61, 68 |
| src/xngin/apiserver/dependencies.py                                  |       12 |        4 |     67% | 12, 23-25 |
| src/xngin/apiserver/dns/safe\_resolve.py                             |       50 |       12 |     76% |32-36, 40-41, 54, 76, 79, 84-85 |
| src/xngin/apiserver/dwh/dwh\_session.py                              |      189 |       61 |     68% |75, 78, 141, 147, 158, 160-161, 172-229, 332-337, 341, 348-350, 369, 371-372, 383, 418-425, 433 |
| src/xngin/apiserver/dwh/dwh\_test\_support.py                        |       84 |        1 |     99% |       175 |
| src/xngin/apiserver/dwh/inspection\_types.py                         |       59 |        5 |     92% |27, 45, 75, 86, 92 |
| src/xngin/apiserver/dwh/inspections.py                               |       64 |        3 |     95% |69, 102-105 |
| src/xngin/apiserver/dwh/participant\_metrics\_queries.py             |      143 |        7 |     95% |73-79, 166, 263, 265, 289 |
| src/xngin/apiserver/dwh/queries.py                                   |       53 |       22 |     58% |   105-149 |
| src/xngin/apiserver/dwh/query\_constructors.py                       |       81 |        6 |     93% |37-39, 104-105, 124-125 |
| src/xngin/apiserver/dwh/test\_dialect\_sql.py                        |       74 |        6 |     92% |544, 557, 560-563 |
| src/xngin/apiserver/dwh/test\_query\_constructors.py                 |      135 |        2 |     99% |  453, 470 |
| src/xngin/apiserver/exceptionhandlers.py                             |       68 |       10 |     85% |50, 54, 62, 70-75, 89, 108 |
| src/xngin/apiserver/flags.py                                         |       49 |        4 |     92% |67, 70, 88, 91 |
| src/xngin/apiserver/main.py                                          |       36 |        5 |     86% |39, 69-70, 87-89 |
| src/xngin/apiserver/openapi.py                                       |       22 |       13 |     41% |     22-86 |
| src/xngin/apiserver/pagination.py                                    |      108 |       13 |     88% |47-48, 53-55, 61, 68, 220, 222, 233-236 |
| src/xngin/apiserver/request\_encapsulation\_middleware.py            |       69 |        3 |     96% |   113-115 |
| src/xngin/apiserver/routers/admin/admin\_api.py                      |      631 |       50 |     92% |254, 353, 365, 376, 380, 389, 578, 803, 807, 866-872, 1055, 1092, 1209-1287, 1334, 1336-1339, 1379, 1501, 1506, 1566, 1579-1580, 1605, 1876, 1882, 1927 |
| src/xngin/apiserver/routers/admin/admin\_api\_converters.py          |       62 |        9 |     85% |29, 73-74, 84, 110-111, 118, 123-124 |
| src/xngin/apiserver/routers/admin/admin\_api\_types.py               |      126 |        2 |     98% |    39, 41 |
| src/xngin/apiserver/routers/admin/test\_admin.py                     |     1351 |        1 |     99% |       361 |
| src/xngin/apiserver/routers/assignment\_adapters.py                  |       42 |        1 |     98% |       145 |
| src/xngin/apiserver/routers/auth/auth\_api.py                        |       61 |       33 |     46% |33-36, 66-77, 83-107, 112-149 |
| src/xngin/apiserver/routers/auth/auth\_dependencies.py               |      137 |       32 |     77% |99, 112-118, 125-152, 249, 257, 270-272 |
| src/xngin/apiserver/routers/auth/test\_auth\_dependencies.py         |      159 |        7 |     96% | 48, 55-61 |
| src/xngin/apiserver/routers/auth/token\_cryptor.py                   |       43 |        4 |     91% |16-17, 53-54 |
| src/xngin/apiserver/routers/common\_api\_types.py                    |      308 |       27 |     91% |122, 148, 150, 426, 428, 430, 487, 713-720, 758, 975, 984, 987-988, 998, 1000, 1010, 1012, 1098, 1253, 1414, 1416-1418 |
| src/xngin/apiserver/routers/common\_enums.py                         |      196 |       40 |     80% |73, 75, 98-107, 112, 134-150, 172-173, 192-196, 237, 272, 288-291, 300, 339-340, 344, 376 |
| src/xngin/apiserver/routers/experiments/dependencies.py              |       61 |        3 |     95% |39, 60, 67 |
| src/xngin/apiserver/routers/experiments/experiments\_common.py       |      410 |       38 |     91% |143, 166, 209, 216, 218, 236, 258, 283-284, 310, 394, 408, 435, 447, 615-616, 690-691, 712, 747, 818, 822, 826, 883-884, 887, 917, 935, 943, 953, 957, 1006-1007, 1015-1017, 1047-1048 |
| src/xngin/apiserver/routers/experiments/experiments\_common\_csv.py  |       32 |        1 |     97% |        66 |
| src/xngin/apiserver/routers/experiments/property\_filters.py         |       96 |        8 |     92% |25, 28, 32, 95-96, 148, 160-161 |
| src/xngin/apiserver/routers/experiments/test\_experiments\_api.py    |      285 |        1 |     99% |        58 |
| src/xngin/apiserver/routers/experiments/test\_experiments\_common.py |      932 |        9 |     99% |186-187, 1119-1121, 1372-1374, 1999 |
| src/xngin/apiserver/routers/experiments/test\_property\_filters.py   |       41 |        1 |     98% |        24 |
| src/xngin/apiserver/routers/healthchecks\_api.py                     |       16 |        2 |     88% |     26-27 |
| src/xngin/apiserver/routers/test\_assignment\_adapters.py            |      198 |        1 |     99% |       102 |
| src/xngin/apiserver/settings.py                                      |      143 |       23 |     84% |101, 121, 128, 134, 179-180, 240, 245, 251-252, 303-307, 326, 348, 359, 361, 371, 374, 391, 416 |
| src/xngin/apiserver/snapshots/fake\_data.py                          |      126 |       37 |     71% |73-81, 87, 90, 92, 97, 102, 107, 199-202, 280, 301-306, 322-349 |
| src/xngin/apiserver/snapshots/snapshotter.py                         |       76 |       23 |     70% |101-102, 114-139, 167-190, 203 |
| src/xngin/apiserver/sql/queries.py                                   |       38 |       10 |     74% | 22, 60-69 |
| src/xngin/apiserver/sqla/tables.py                                   |      304 |        3 |     99% |155, 355, 359 |
| src/xngin/apiserver/storage/bootstrap.py                             |       41 |        1 |     98% |        58 |
| src/xngin/apiserver/storage/storage\_format\_converters.py           |      155 |        8 |     95% |54-56, 268, 303, 443, 482-483 |
| src/xngin/apiserver/testing/admin\_api\_client.py                    |      186 |       14 |     92% |158-161, 206, 223-230, 255-259, 315, 1655, 2601, 3819 |
| src/xngin/apiserver/testing/assertions.py                            |        7 |        1 |     86% |         7 |
| src/xngin/apiserver/testing/experiments\_api\_client.py              |       96 |       11 |     89% |164, 181-188, 202-203, 213-217, 224, 535 |
| src/xngin/apiserver/testing/pg\_helpers.py                           |       13 |        2 |     85% |     16-17 |
| src/xngin/db\_extensions/custom\_functions.py                        |       39 |        3 |     92% |47, 66, 79 |
| src/xngin/db\_extensions/test\_custom\_functions.py                  |       46 |       12 |     74% |59-68, 73-82 |
| src/xngin/events/common.py                                           |        7 |        2 |     71% |    14, 21 |
| src/xngin/events/experiment\_created.py                              |       13 |        1 |     92% |        24 |
| src/xngin/events/webhook\_sent.py                                    |       12 |        5 |     58% |     18-23 |
| src/xngin/ops/performance.py                                         |       17 |        1 |     94% |        23 |
| src/xngin/ops/sentry.py                                              |       13 |        6 |     54% |     18-40 |
| src/xngin/stats/assignment.py                                        |       58 |        1 |     98% |       120 |
| src/xngin/stats/balance.py                                           |       67 |        2 |     97% |  110, 138 |
| src/xngin/stats/bandit\_analysis.py                                  |       75 |        5 |     93% |135, 137, 139, 202-203 |
| src/xngin/stats/bandit\_sampling.py                                  |       73 |        7 |     90% |162, 186, 191, 214, 238, 240, 272 |
| src/xngin/stats/power.py                                             |      118 |        5 |     96% |71, 74, 118-119, 195 |
| src/xngin/xsecrets/chafernet.py                                      |       52 |        1 |     98% |        92 |
| src/xngin/xsecrets/gcp\_kms\_provider.py                             |       70 |       28 |     60% |64-79, 86-87, 104-108, 111, 115-123, 127-134 |
| src/xngin/xsecrets/provider.py                                       |       19 |        1 |     95% |        46 |
| src/xngin/xsecrets/secretservice.py                                  |       60 |        4 |     93% |41-42, 100, 122 |
| src/xngin/xsecrets/test\_gcp\_kms\_provider.py                       |      103 |       26 |     75% |40-42, 170-175, 182-189, 195-199, 206, 213-224 |
| src/xngin/xsecrets/test\_nacl\_provider.py                           |       67 |        1 |     99% |        24 |
| **TOTAL**                                                            | **10791** |  **735** | **93%** |           |

60 files skipped due to complete coverage.


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