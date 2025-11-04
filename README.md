# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/agency-fund/evidential-be/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                 |    Stmts |     Miss |   Cover |   Missing |
|--------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/xngin/apiserver/apikeys.py                                       |       39 |        8 |     79% |35, 37-41, 70-73 |
| src/xngin/apiserver/certs/certs.py                                   |       16 |        9 |     44% |19-23, 37-44 |
| src/xngin/apiserver/common\_field\_types.py                          |       12 |        1 |     92% |        13 |
| src/xngin/apiserver/conftest.py                                      |      186 |        8 |     96% |74, 96-98, 108, 157, 159, 163 |
| src/xngin/apiserver/customlogging.py                                 |       62 |       10 |     84% |24-25, 47, 64-65, 93-98 |
| src/xngin/apiserver/database.py                                      |       49 |        5 |     90% |27, 38, 55, 61, 68 |
| src/xngin/apiserver/dependencies.py                                  |       12 |        4 |     67% | 12, 23-25 |
| src/xngin/apiserver/dns/safe\_resolve.py                             |       47 |       12 |     74% |28-32, 36-37, 50, 72, 75, 80-81 |
| src/xngin/apiserver/dwh/dwh\_session.py                              |      178 |       75 |     58% |36, 42, 70-77, 80, 143, 149, 160, 162-163, 165-167, 174-231, 251-253, 268, 334-339, 342, 344-350, 368, 401-408, 416 |
| src/xngin/apiserver/dwh/inspection\_types.py                         |       61 |        5 |     92% |27, 45, 67, 78, 84 |
| src/xngin/apiserver/dwh/inspections.py                               |       28 |        1 |     96% |        63 |
| src/xngin/apiserver/dwh/queries.py                                   |      194 |       33 |     83% |118-158, 170, 177, 217, 247-249, 299, 303, 356-357, 366, 368, 377-378, 387-388 |
| src/xngin/apiserver/dwh/test\_dialect\_sql.py                        |       79 |        6 |     92% |560, 573, 576-579 |
| src/xngin/apiserver/dwh/test\_queries.py                             |      237 |        1 |     99% |       212 |
| src/xngin/apiserver/exceptionhandlers.py                             |       49 |        9 |     82% |31, 35, 43, 51-56, 70 |
| src/xngin/apiserver/flags.py                                         |       41 |        2 |     95% |    72, 75 |
| src/xngin/apiserver/main.py                                          |       29 |        1 |     97% |        38 |
| src/xngin/apiserver/openapi.py                                       |       24 |       13 |     46% |     19-71 |
| src/xngin/apiserver/request\_encapsulation\_middleware.py            |       63 |        3 |     95% |   103-105 |
| src/xngin/apiserver/routers/admin/admin\_api.py                      |      558 |      291 |     48% |173, 203-206, 238-241, 266-269, 283-287, 299, 310, 314, 323, 340, 356-367, 387-400, 441-453, 470, 492, 503, 518-531, 551-561, 593-608, 626-641, 676-688, 692-705, 725-741, 787-793, 810-818, 847-858, 883-893, 904-923, 934-935, 954-961, 967-973, 993-1004, 1021, 1071-1081, 1097-1183, 1195, 1210-1231, 1256-1262, 1316-1321, 1374-1383, 1413-1438, 1455-1473, 1500-1501, 1516-1517, 1528, 1542-1548, 1559-1565, 1582-1589, 1624-1626, 1644-1667, 1707-1710, 1713-1722, 1733-1759, 1770-1774 |
| src/xngin/apiserver/routers/admin/admin\_api\_converters.py          |       62 |       16 |     74% |29, 73-74, 84, 110-111, 118, 123-124, 138-144 |
| src/xngin/apiserver/routers/admin/admin\_api\_types.py               |      214 |        2 |     99% |    36, 38 |
| src/xngin/apiserver/routers/admin/generic\_handlers.py               |       24 |       14 |     42% |     42-60 |
| src/xngin/apiserver/routers/admin/test\_admin.py                     |     1133 |        1 |     99% |       310 |
| src/xngin/apiserver/routers/assignment\_adapters.py                  |       41 |        1 |     98% |       141 |
| src/xngin/apiserver/routers/auth/auth\_api.py                        |       62 |       33 |     47% |34-37, 67-78, 84-108, 113-150 |
| src/xngin/apiserver/routers/auth/auth\_dependencies.py               |      125 |       32 |     74% |76, 89-95, 102-129, 222, 230, 243-245 |
| src/xngin/apiserver/routers/auth/session\_token\_crypter.py          |       43 |        9 |     79% |20-25, 49, 51-52 |
| src/xngin/apiserver/routers/auth/test\_auth\_dependencies.py         |      125 |        1 |     99% |        48 |
| src/xngin/apiserver/routers/common\_api\_types.py                    |      357 |       24 |     93% |111, 137, 139, 350, 352, 354, 395, 629-631, 669, 870, 879, 882-883, 893, 895, 905, 907, 1109, 1244, 1246-1248 |
| src/xngin/apiserver/routers/common\_enums.py                         |      176 |       40 |     77% |66, 68, 80, 82, 91-100, 105, 127-143, 154-158, 199, 234, 250-253, 262, 283-284, 288, 320 |
| src/xngin/apiserver/routers/experiments/dependencies.py              |       37 |        8 |     78% |43, 46-50, 103-109 |
| src/xngin/apiserver/routers/experiments/experiments\_api.py          |       53 |       14 |     74% |96, 110, 174, 187-201, 220-230 |
| src/xngin/apiserver/routers/experiments/experiments\_common.py       |      351 |       50 |     86% |77, 102, 122, 129, 162-163, 187, 266, 280, 316, 353-369, 401-402, 490-491, 549-551, 622-623, 661-675, 708, 712, 716, 720, 778-779, 782, 812, 830, 838, 848, 852, 898-899, 905-907, 929-938 |
| src/xngin/apiserver/routers/experiments/test\_experiments\_common.py |      519 |        6 |     99% |175-176, 582-584, 1216 |
| src/xngin/apiserver/routers/healthchecks\_api.py                     |       16 |        2 |     88% |     26-27 |
| src/xngin/apiserver/routers/test\_assignment\_adapters.py            |      198 |        1 |     99% |       111 |
| src/xngin/apiserver/settings.py                                      |      171 |       28 |     84% |99, 119, 126, 132, 177-178, 238, 243, 246-251, 301-305, 324, 346, 357, 359, 369, 372, 389, 410, 425 |
| src/xngin/apiserver/snapshots/snapshotter.py                         |       70 |       32 |     54% |28-64, 92-95, 106-128, 152-175, 187 |
| src/xngin/apiserver/sqla/tables.py                                   |      270 |        6 |     98% |152-154, 300, 334-335 |
| src/xngin/apiserver/storage/storage\_format\_converters.py           |      105 |        7 |     93% |58, 89-90, 171, 206, 359, 372 |
| src/xngin/apiserver/testing/assertions.py                            |       14 |        5 |     64% | 14-17, 23 |
| src/xngin/apiserver/testing/xurl.py                                  |       29 |        1 |     97% |        37 |
| src/xngin/db\_extensions/custom\_functions.py                        |       39 |        3 |     92% |47, 66, 79 |
| src/xngin/db\_extensions/test\_custom\_functions.py                  |       48 |       12 |     75% |59-68, 73-82 |
| src/xngin/events/common.py                                           |        7 |        2 |     71% |    14, 21 |
| src/xngin/events/experiment\_created.py                              |       14 |        4 |     71% | 18, 21-25 |
| src/xngin/events/webhook\_sent.py                                    |       15 |        5 |     67% |     18-23 |
| src/xngin/ops/sentry.py                                              |       13 |        6 |     54% |     18-40 |
| src/xngin/stats/assignment.py                                        |       45 |        1 |     98% |       112 |
| src/xngin/stats/balance.py                                           |       69 |        2 |     97% |  106, 145 |
| src/xngin/stats/bandit\_analysis.py                                  |       56 |        5 |     91% |94, 96, 98, 157-158 |
| src/xngin/stats/bandit\_sampling.py                                  |       73 |        7 |     90% |159, 183, 188, 211, 235, 237, 269 |
| src/xngin/stats/power.py                                             |       65 |        5 |     92% |44, 83, 140-150 |
| src/xngin/xsecrets/chafernet.py                                      |       53 |        1 |     98% |        93 |
| src/xngin/xsecrets/gcp\_kms\_provider.py                             |       71 |       26 |     63% |62-77, 84-85, 102-103, 106, 110-118, 122-129 |
| src/xngin/xsecrets/provider.py                                       |       19 |        1 |     95% |        46 |
| src/xngin/xsecrets/secretservice.py                                  |       61 |        4 |     93% |42-43, 101, 123 |
| src/xngin/xsecrets/test\_gcp\_kms\_provider.py                       |      103 |       26 |     75% |40-42, 170-175, 182-189, 195-199, 206, 213-224 |
| src/xngin/xsecrets/test\_nacl\_provider.py                           |       67 |        1 |     99% |        24 |
|                                                            **TOTAL** | **8251** |  **941** | **89%** |           |

56 files skipped due to complete coverage.


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