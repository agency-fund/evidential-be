# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/agency-fund/evidential-be/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                 |    Stmts |     Miss |   Cover |   Missing |
|--------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/xngin/apiserver/apikeys.py                                       |       39 |        8 |     79% |35, 37-41, 70-73 |
| src/xngin/apiserver/certs/certs.py                                   |       16 |        9 |     44% |19-23, 37-44 |
| src/xngin/apiserver/common\_field\_types.py                          |       12 |        1 |     92% |        13 |
| src/xngin/apiserver/conftest.py                                      |      181 |        7 |     96% |71, 86, 88, 99, 150, 152, 156 |
| src/xngin/apiserver/customlogging.py                                 |       66 |       13 |     80% |25-26, 48-68, 73-74, 102-107 |
| src/xngin/apiserver/database.py                                      |       49 |        5 |     90% |27, 38, 55, 61, 68 |
| src/xngin/apiserver/dependencies.py                                  |       12 |        4 |     67% | 12, 23-25 |
| src/xngin/apiserver/dns/safe\_resolve.py                             |       47 |       12 |     74% |28-32, 36-37, 50, 72, 75, 80-81 |
| src/xngin/apiserver/dwh/dwh\_session.py                              |      178 |       75 |     58% |36, 42, 70-77, 80, 143, 149, 160, 162-163, 165-167, 174-231, 251-253, 268, 334-339, 342, 344-350, 368, 401-408, 416 |
| src/xngin/apiserver/dwh/inspection\_types.py                         |       61 |        5 |     92% |27, 45, 67, 78, 84 |
| src/xngin/apiserver/dwh/inspections.py                               |       28 |        2 |     93% |    63, 90 |
| src/xngin/apiserver/dwh/queries.py                                   |      152 |       27 |     82% |116-156, 168, 175, 215, 243-245, 296-297, 316-317 |
| src/xngin/apiserver/dwh/test\_dialect\_sql.py                        |       79 |        6 |     92% |560, 573, 576-579 |
| src/xngin/apiserver/dwh/test\_queries.py                             |      280 |        3 |     99% |214, 668, 685 |
| src/xngin/apiserver/exceptionhandlers.py                             |       49 |        9 |     82% |31, 35, 43, 51-56, 70 |
| src/xngin/apiserver/flags.py                                         |       49 |        4 |     92% |67, 70, 88, 91 |
| src/xngin/apiserver/main.py                                          |       36 |        5 |     86% |39, 66-67, 84-86 |
| src/xngin/apiserver/openapi.py                                       |       24 |       13 |     46% |     19-79 |
| src/xngin/apiserver/request\_encapsulation\_middleware.py            |       69 |        3 |     96% |   113-115 |
| src/xngin/apiserver/routers/admin/admin\_api.py                      |      559 |      283 |     49% |173, 203-206, 238-241, 266-269, 283-287, 299, 310, 314, 323, 340, 356-367, 387-400, 441-453, 470, 492, 500, 515-528, 548-558, 590-605, 623-638, 673-685, 689-702, 722-738, 784-790, 807-815, 844-855, 880-887, 898-917, 928-929, 948-955, 961-967, 987-998, 1015, 1065-1075, 1091-1175, 1187, 1202-1223, 1248-1254, 1308-1313, 1366-1375, 1405-1430, 1447-1465, 1492-1493, 1508-1509, 1520, 1534-1540, 1551-1557, 1574-1581, 1616-1618, 1637-1663, 1703-1706, 1709-1718, 1731, 1736-1744, 1774 |
| src/xngin/apiserver/routers/admin/admin\_api\_converters.py          |       62 |       16 |     74% |29, 73-74, 84, 110-111, 118, 123-124, 138-144 |
| src/xngin/apiserver/routers/admin/admin\_api\_types.py               |      216 |        2 |     99% |    37, 39 |
| src/xngin/apiserver/routers/admin/generic\_handlers.py               |       24 |       14 |     42% |     42-60 |
| src/xngin/apiserver/routers/admin/test\_admin.py                     |     1251 |        1 |     99% |       387 |
| src/xngin/apiserver/routers/assignment\_adapters.py                  |       41 |        1 |     98% |       144 |
| src/xngin/apiserver/routers/auth/auth\_api.py                        |       62 |       33 |     47% |34-37, 67-78, 84-108, 113-150 |
| src/xngin/apiserver/routers/auth/auth\_dependencies.py               |      125 |       32 |     74% |76, 89-95, 102-129, 222, 230, 243-245 |
| src/xngin/apiserver/routers/auth/session\_token\_crypter.py          |       43 |        9 |     79% |20-25, 49, 51-52 |
| src/xngin/apiserver/routers/auth/test\_auth\_dependencies.py         |      128 |        1 |     99% |        48 |
| src/xngin/apiserver/routers/common\_api\_types.py                    |      388 |       24 |     94% |113, 139, 141, 417, 419, 421, 478, 709-711, 749, 957, 966, 969-970, 980, 982, 992, 994, 1196, 1351, 1353-1355 |
| src/xngin/apiserver/routers/common\_enums.py                         |      183 |       38 |     79% |66, 68, 91-100, 105, 127-143, 154-158, 199, 234, 250-253, 262, 301-302, 306, 338 |
| src/xngin/apiserver/routers/experiments/dependencies.py              |       38 |        8 |     79% |43, 46-50, 103-109 |
| src/xngin/apiserver/routers/experiments/experiments\_api.py          |       58 |       14 |     76% |98, 112, 217, 230-244, 263-273 |
| src/xngin/apiserver/routers/experiments/experiments\_common.py       |      390 |       55 |     86% |82-89, 131, 150, 186-187, 210, 288, 299, 326, 338, 376-392, 424-425, 513-514, 572-574, 645-646, 667, 700-715, 769, 773, 777, 781, 838-839, 842, 872, 890, 898, 908, 912, 961-962, 970-972, 1001-1003 |
| src/xngin/apiserver/routers/experiments/property\_filters.py         |       96 |        8 |     92% |25, 28, 32, 95-96, 148, 160-161 |
| src/xngin/apiserver/routers/experiments/test\_experiments\_common.py |      719 |        6 |     99% |178-179, 850-852, 1579 |
| src/xngin/apiserver/routers/experiments/test\_property\_filters.py   |       44 |        1 |     98% |        24 |
| src/xngin/apiserver/routers/healthchecks\_api.py                     |       16 |        2 |     88% |     26-27 |
| src/xngin/apiserver/routers/test\_assignment\_adapters.py            |      196 |        1 |     99% |       109 |
| src/xngin/apiserver/settings.py                                      |      164 |       27 |     84% |99, 119, 126, 132, 177-178, 238, 243, 246-251, 301-305, 324, 346, 357, 359, 369, 372, 389, 414 |
| src/xngin/apiserver/snapshots/snapshotter.py                         |       75 |       23 |     69% |94-95, 107-129, 157-180, 192 |
| src/xngin/apiserver/sqla/tables.py                                   |      274 |        6 |     98% |152-154, 300, 334-335 |
| src/xngin/apiserver/storage/bootstrap.py                             |       34 |        1 |     97% |        34 |
| src/xngin/apiserver/storage/storage\_format\_converters.py           |      111 |        8 |     93% |58, 89-90, 175, 210, 346, 385-386 |
| src/xngin/apiserver/testing/assertions.py                            |       14 |        5 |     64% | 14-17, 23 |
| src/xngin/apiserver/testing/xurl.py                                  |       29 |        1 |     97% |        37 |
| src/xngin/db\_extensions/custom\_functions.py                        |       39 |        3 |     92% |47, 66, 79 |
| src/xngin/db\_extensions/test\_custom\_functions.py                  |       48 |       12 |     75% |59-68, 73-82 |
| src/xngin/events/common.py                                           |        7 |        2 |     71% |    14, 21 |
| src/xngin/events/experiment\_created.py                              |       14 |        4 |     71% | 18, 21-25 |
| src/xngin/events/webhook\_sent.py                                    |       15 |        5 |     67% |     18-23 |
| src/xngin/ops/sentry.py                                              |       13 |        6 |     54% |     18-40 |
| src/xngin/stats/assignment.py                                        |       62 |        1 |     98% |       120 |
| src/xngin/stats/balance.py                                           |       71 |        2 |     97% |  109, 148 |
| src/xngin/stats/bandit\_analysis.py                                  |       75 |        5 |     93% |135, 137, 139, 202-203 |
| src/xngin/stats/bandit\_sampling.py                                  |       73 |        7 |     90% |162, 186, 191, 214, 238, 240, 272 |
| src/xngin/stats/power.py                                             |       75 |        4 |     95% |89, 163-174 |
| src/xngin/xsecrets/chafernet.py                                      |       52 |        1 |     98% |        92 |
| src/xngin/xsecrets/gcp\_kms\_provider.py                             |       71 |       26 |     63% |62-77, 84-85, 102-103, 106, 110-118, 122-129 |
| src/xngin/xsecrets/provider.py                                       |       19 |        1 |     95% |        46 |
| src/xngin/xsecrets/secretservice.py                                  |       61 |        4 |     93% |42-43, 101, 123 |
| src/xngin/xsecrets/test\_gcp\_kms\_provider.py                       |      103 |       26 |     75% |40-42, 170-175, 182-189, 195-199, 206, 213-224 |
| src/xngin/xsecrets/test\_nacl\_provider.py                           |       67 |        1 |     99% |        24 |
|                                                            **TOTAL** | **9207** |  **941** | **90%** |           |

57 files skipped due to complete coverage.


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