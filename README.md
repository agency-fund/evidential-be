# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/agency-fund/evidential-be/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                 |    Stmts |     Miss |   Cover |   Missing |
|--------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/xngin/apiserver/apikeys.py                                       |       51 |        7 |     86% |59-63, 92-95 |
| src/xngin/apiserver/certs/certs.py                                   |       16 |        9 |     44% |19-23, 37-44 |
| src/xngin/apiserver/common\_field\_types.py                          |       12 |        1 |     92% |        13 |
| src/xngin/apiserver/conftest.py                                      |      181 |        7 |     96% |70, 85, 87, 98, 149, 151, 155 |
| src/xngin/apiserver/customlogging.py                                 |       66 |       13 |     80% |25-26, 48-68, 73-74, 102-107 |
| src/xngin/apiserver/database.py                                      |       49 |        5 |     90% |27, 38, 55, 61, 68 |
| src/xngin/apiserver/dependencies.py                                  |       12 |        4 |     67% | 12, 23-25 |
| src/xngin/apiserver/dns/safe\_resolve.py                             |       50 |       12 |     76% |32-36, 40-41, 54, 76, 79, 84-85 |
| src/xngin/apiserver/dwh/dwh\_session.py                              |      192 |       61 |     68% |74, 77, 140, 146, 157, 159-160, 171-228, 331-336, 340, 347-349, 368, 370-371, 382, 417-424, 432 |
| src/xngin/apiserver/dwh/inspection\_types.py                         |       63 |        5 |     92% |27, 45, 75, 86, 92 |
| src/xngin/apiserver/dwh/inspections.py                               |       64 |        3 |     95% |69, 102-105 |
| src/xngin/apiserver/dwh/queries.py                                   |      156 |       31 |     80% |118-162, 174, 181, 221, 249-251, 302-303, 322-323 |
| src/xngin/apiserver/dwh/test\_dialect\_sql.py                        |       79 |        6 |     92% |560, 573, 576-579 |
| src/xngin/apiserver/dwh/test\_queries.py                             |      280 |        3 |     99% |214, 668, 685 |
| src/xngin/apiserver/exceptionhandlers.py                             |       63 |       10 |     84% |33, 37, 45, 53-58, 72, 91 |
| src/xngin/apiserver/flags.py                                         |       49 |        4 |     92% |67, 70, 88, 91 |
| src/xngin/apiserver/main.py                                          |       36 |        5 |     86% |39, 69-70, 87-89 |
| src/xngin/apiserver/openapi.py                                       |       24 |       13 |     46% |     19-83 |
| src/xngin/apiserver/pagination.py                                    |      112 |       13 |     88% |47-48, 53-55, 61, 68, 220, 222, 233-236 |
| src/xngin/apiserver/request\_encapsulation\_middleware.py            |       69 |        3 |     96% |   113-115 |
| src/xngin/apiserver/routers/admin/admin\_api.py                      |      618 |      316 |     49% |203, 223-226, 258-261, 286-289, 303-307, 319, 330, 334, 343, 360, 376-387, 408-434, 463-464, 481-493, 510, 532, 540, 554-557, 577-587, 619-634, 652-667, 692-693, 705-719, 756-772, 804-805, 820-826, 843-851, 880-901, 912, 928-931, 935-938, 949-968, 979-980, 999-1003, 1006, 1011, 1031-1048, 1065, 1093-1094, 1104, 1124-1134, 1151-1229, 1241, 1267-1292, 1317-1323, 1338-1339, 1379-1384, 1413-1414, 1439-1463, 1485-1510, 1527-1545, 1572-1575, 1590-1593, 1604, 1618-1625, 1639-1645, 1662-1669, 1704-1706, 1725-1751, 1773-1774, 1790-1802, 1818-1821, 1824-1833, 1846, 1851-1866, 1897 |
| src/xngin/apiserver/routers/admin/admin\_api\_converters.py          |       62 |        9 |     85% |29, 73-74, 84, 110-111, 118, 123-124 |
| src/xngin/apiserver/routers/admin/admin\_api\_types.py               |      246 |        2 |     99% |    39, 41 |
| src/xngin/apiserver/routers/admin/generic\_handlers.py               |       23 |       13 |     43% |     42-59 |
| src/xngin/apiserver/routers/admin/test\_admin.py                     |     1678 |        1 |     99% |       399 |
| src/xngin/apiserver/routers/assignment\_adapters.py                  |       41 |        1 |     98% |       144 |
| src/xngin/apiserver/routers/auth/auth\_api.py                        |       61 |       33 |     46% |33-36, 66-77, 83-107, 112-149 |
| src/xngin/apiserver/routers/auth/auth\_dependencies.py               |      135 |       32 |     76% |99, 112-118, 125-152, 240, 248, 261-263 |
| src/xngin/apiserver/routers/auth/test\_auth\_dependencies.py         |      152 |        7 |     95% | 47, 54-60 |
| src/xngin/apiserver/routers/auth/token\_cryptor.py                   |       43 |        4 |     91% |16-17, 53-54 |
| src/xngin/apiserver/routers/common\_api\_types.py                    |      405 |       25 |     94% |122, 148, 150, 426, 428, 430, 487, 718-720, 758, 975, 984, 987-988, 998, 1000, 1010, 1012, 1098, 1253, 1414, 1416-1418 |
| src/xngin/apiserver/routers/common\_enums.py                         |      183 |       38 |     79% |66, 68, 91-100, 105, 127-143, 154-158, 199, 234, 250-253, 262, 301-302, 306, 338 |
| src/xngin/apiserver/routers/experiments/dependencies.py              |       51 |        9 |     82% |44, 65, 68-72, 125-131 |
| src/xngin/apiserver/routers/experiments/experiments\_api.py          |       68 |       18 |     74% |122, 136, 185-198, 269, 282-296, 325 |
| src/xngin/apiserver/routers/experiments/experiments\_common.py       |      423 |       59 |     86% |83-90, 131, 154, 197, 201, 220, 239, 260-261, 284, 362, 373, 400, 412, 450-466, 496-497, 585-586, 644-646, 717-718, 739, 772-787, 841, 845, 849, 853, 910-911, 914, 944, 962, 970, 980, 984, 1033-1034, 1042-1044, 1073-1075 |
| src/xngin/apiserver/routers/experiments/property\_filters.py         |       96 |        8 |     92% |25, 28, 32, 95-96, 148, 160-161 |
| src/xngin/apiserver/routers/experiments/test\_experiments\_common.py |      723 |        6 |     99% |179-180, 851-853, 1584 |
| src/xngin/apiserver/routers/experiments/test\_property\_filters.py   |       44 |        1 |     98% |        24 |
| src/xngin/apiserver/routers/healthchecks\_api.py                     |       16 |        2 |     88% |     26-27 |
| src/xngin/apiserver/routers/test\_assignment\_adapters.py            |      196 |        1 |     99% |       109 |
| src/xngin/apiserver/settings.py                                      |      163 |       23 |     86% |101, 121, 128, 134, 179-180, 240, 245, 251-252, 303-307, 326, 348, 359, 361, 371, 374, 391, 416 |
| src/xngin/apiserver/snapshots/snapshotter.py                         |       75 |       23 |     69% |94-95, 107-129, 157-180, 192 |
| src/xngin/apiserver/sqla/tables.py                                   |      274 |        4 |     99% |153, 300, 334-335 |
| src/xngin/apiserver/storage/bootstrap.py                             |       37 |        1 |     97% |        52 |
| src/xngin/apiserver/storage/storage\_format\_converters.py           |      111 |        8 |     93% |58, 89-90, 175, 210, 346, 385-386 |
| src/xngin/apiserver/testing/assertions.py                            |       14 |        5 |     64% | 14-17, 23 |
| src/xngin/apiserver/testing/xurl.py                                  |       29 |        1 |     97% |        37 |
| src/xngin/db\_extensions/custom\_functions.py                        |       39 |        3 |     92% |47, 66, 79 |
| src/xngin/db\_extensions/test\_custom\_functions.py                  |       48 |       12 |     75% |59-68, 73-82 |
| src/xngin/events/common.py                                           |        7 |        2 |     71% |    14, 21 |
| src/xngin/events/experiment\_created.py                              |       14 |        1 |     93% |        24 |
| src/xngin/events/webhook\_sent.py                                    |       15 |        5 |     67% |     18-23 |
| src/xngin/ops/sentry.py                                              |       13 |        6 |     54% |     18-40 |
| src/xngin/stats/assignment.py                                        |       62 |        1 |     98% |       120 |
| src/xngin/stats/balance.py                                           |       71 |        2 |     97% |  109, 148 |
| src/xngin/stats/bandit\_analysis.py                                  |       75 |        5 |     93% |135, 137, 139, 202-203 |
| src/xngin/stats/bandit\_sampling.py                                  |       73 |        7 |     90% |162, 186, 191, 214, 238, 240, 272 |
| src/xngin/stats/power.py                                             |      118 |        5 |     96% |71, 74, 118-119, 195 |
| src/xngin/xsecrets/chafernet.py                                      |       52 |        1 |     98% |        92 |
| src/xngin/xsecrets/gcp\_kms\_provider.py                             |       71 |       26 |     63% |62-77, 84-85, 102-103, 106, 110-118, 122-129 |
| src/xngin/xsecrets/provider.py                                       |       19 |        1 |     95% |        46 |
| src/xngin/xsecrets/secretservice.py                                  |       61 |        4 |     93% |42-43, 101, 123 |
| src/xngin/xsecrets/test\_gcp\_kms\_provider.py                       |      103 |       26 |     75% |40-42, 170-175, 182-189, 195-199, 206, 213-224 |
| src/xngin/xsecrets/test\_nacl\_provider.py                           |       67 |        1 |     99% |        24 |
|                                                            **TOTAL** | **10407** |  **973** | **91%** |           |

59 files skipped due to complete coverage.


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