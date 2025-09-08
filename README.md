# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/agency-fund/evidential-be/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                 |    Stmts |     Miss |   Cover |   Missing |
|--------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/xngin/apiserver/apikeys.py                                       |       39 |        8 |     79% |35, 37-41, 70-73 |
| src/xngin/apiserver/certs/certs.py                                   |       16 |        9 |     44% |19-23, 37-44 |
| src/xngin/apiserver/common\_field\_types.py                          |       12 |        1 |     92% |        13 |
| src/xngin/apiserver/conftest.py                                      |      186 |        8 |     96% |74, 96-98, 108, 157, 159, 163 |
| src/xngin/apiserver/customlogging.py                                 |       54 |        5 |     91% |35-36, 41, 56-57 |
| src/xngin/apiserver/database.py                                      |       62 |       16 |     74% |28, 39, 56, 62, 69, 90-102 |
| src/xngin/apiserver/dependencies.py                                  |       27 |        9 |     67% |22, 49, 52-56, 61-63 |
| src/xngin/apiserver/dns/safe\_resolve.py                             |       44 |        9 |     80% |28-29, 33-34, 47, 69, 72, 77-78 |
| src/xngin/apiserver/dwh/dwh\_session.py                              |      179 |       75 |     58% |37, 43, 71-78, 81, 144, 150, 161, 163-164, 166-168, 175-232, 254-256, 271, 337-342, 345, 347-353, 371, 405-412, 420 |
| src/xngin/apiserver/dwh/inspection\_types.py                         |       61 |        5 |     92% |27, 45, 67, 78, 84 |
| src/xngin/apiserver/dwh/inspections.py                               |       28 |        1 |     96% |        63 |
| src/xngin/apiserver/dwh/queries.py                                   |      178 |       33 |     81% |116-156, 168, 175, 215, 245-247, 293, 314, 331-332, 341, 343, 346-347, 356-357 |
| src/xngin/apiserver/dwh/test\_dialect\_sql.py                        |       78 |        6 |     92% |592, 605, 608-611 |
| src/xngin/apiserver/dwh/test\_queries.py                             |      238 |        1 |     99% |       210 |
| src/xngin/apiserver/exceptionhandlers.py                             |       49 |        9 |     82% |31, 35, 43, 51-56, 70 |
| src/xngin/apiserver/main.py                                          |       29 |        1 |     97% |        37 |
| src/xngin/apiserver/openapi.py                                       |       20 |        9 |     55% |     19-67 |
| src/xngin/apiserver/routers/admin/admin\_api.py                      |      515 |      257 |     50% |174, 237-240, 272-275, 300-303, 317-321, 334-336, 358-369, 389-402, 443-459, 476, 498, 509, 524-537, 557-567, 599-614, 632-647, 682-694, 698-711, 731-747, 793-799, 816-824, 853-864, 889-899, 910-929, 940-941, 960-967, 973-979, 999-1010, 1027, 1077-1087, 1103-1189, 1201, 1216-1237, 1262-1268, 1322-1327, 1380-1389, 1414-1442, 1466-1467, 1481-1482, 1493, 1507-1516, 1527-1533, 1550-1557, 1592-1612, 1651-1677, 1688-1692 |
| src/xngin/apiserver/routers/admin/admin\_api\_converters.py          |       62 |       16 |     74% |29, 73-74, 84, 110-111, 118, 123-124, 138-144 |
| src/xngin/apiserver/routers/admin/admin\_api\_types.py               |      197 |        2 |     99% |    35, 37 |
| src/xngin/apiserver/routers/admin/generic\_handlers.py               |       24 |       14 |     42% |     42-60 |
| src/xngin/apiserver/routers/admin/test\_admin.py                     |      949 |        1 |     99% |       334 |
| src/xngin/apiserver/routers/assignment\_adapters.py                  |       41 |        1 |     98% |       141 |
| src/xngin/apiserver/routers/auth/auth\_api.py                        |       35 |       12 |     66% |29-32, 59-80 |
| src/xngin/apiserver/routers/auth/auth\_dependencies.py               |       96 |       47 |     51% |52, 65-71, 78-105, 123, 130-169, 200-202 |
| src/xngin/apiserver/routers/common\_api\_types.py                    |      350 |       24 |     93% |108, 134, 136, 347, 349, 351, 370, 599-601, 638, 830, 839, 842-843, 853, 855, 865, 867, 1069, 1205, 1207-1209 |
| src/xngin/apiserver/routers/common\_enums.py                         |      174 |       40 |     77% |71, 73, 85, 87, 96-105, 110, 132-149, 160-164, 205, 232, 252-255, 264, 285-286, 290, 322 |
| src/xngin/apiserver/routers/experiments/dependencies.py              |       18 |        3 |     83% |     49-55 |
| src/xngin/apiserver/routers/experiments/experiments\_api.py          |       69 |       26 |     62% |91, 102, 116, 147-154, 193, 207-213, 217-248, 266-276 |
| src/xngin/apiserver/routers/experiments/experiments\_common.py       |      324 |       38 |     88% |70, 75, 95, 115, 122, 155-156, 180, 257, 271, 307, 338-354, 469, 523-525, 596-597, 640, 644, 648, 652, 711-712, 715, 745, 763, 771, 781, 785, 827-828, 834-836 |
| src/xngin/apiserver/routers/experiments/test\_experiments\_common.py |      505 |        6 |     99% |175-176, 570-572, 1186 |
| src/xngin/apiserver/routers/healthchecks\_api.py                     |       16 |        2 |     88% |     26-27 |
| src/xngin/apiserver/routers/test\_assignment\_adapters.py            |      198 |        1 |     99% |       111 |
| src/xngin/apiserver/routers/test\_common\_api\_types.py              |       41 |        1 |     98% |        74 |
| src/xngin/apiserver/settings.py                                      |      171 |       28 |     84% |99, 119, 126, 132, 177-178, 238, 243, 246-251, 301-305, 324, 346, 357, 359, 369, 372, 389, 411, 426 |
| src/xngin/apiserver/snapshots/snapshotter.py                         |       57 |       21 |     63% |27-63, 89-92, 103-123, 147 |
| src/xngin/apiserver/sqla/tables.py                                   |      257 |        6 |     98% |132-134, 262, 296-297 |
| src/xngin/apiserver/storage/storage\_format\_converters.py           |      100 |        7 |     93% |58, 89-90, 170, 205, 356, 369 |
| src/xngin/apiserver/testing/assertions.py                            |       14 |        5 |     64% | 14-17, 23 |
| src/xngin/apiserver/testing/xurl.py                                  |       29 |        1 |     97% |        37 |
| src/xngin/cli/main.py                                                |      336 |      197 |     41% |83-99, 115-146, 150-154, 164-166, 253, 256, 261, 273-274, 283, 291-309, 312-348, 350-353, 371-376, 381-382, 411-412, 425-433, 439-445, 451-454, 461-469, 494-499, 514-522, 535-543, 547-550, 590-666, 682-686, 697-698, 706-708, 712 |
| src/xngin/db\_extensions/custom\_functions.py                        |       39 |        3 |     92% |47, 66, 79 |
| src/xngin/db\_extensions/test\_custom\_functions.py                  |       48 |       12 |     75% |59-68, 73-82 |
| src/xngin/events/common.py                                           |        7 |        2 |     71% |    14, 21 |
| src/xngin/events/experiment\_created.py                              |       14 |        4 |     71% | 18, 21-25 |
| src/xngin/events/webhook\_sent.py                                    |       15 |        5 |     67% |     18-23 |
| src/xngin/ops/sentry.py                                              |       13 |        6 |     54% |     18-40 |
| src/xngin/stats/assignment.py                                        |       45 |        1 |     98% |       111 |
| src/xngin/stats/balance.py                                           |       62 |        3 |     95% |106, 134, 137 |
| src/xngin/stats/bandit\_sampling.py                                  |       73 |        7 |     90% |158, 182, 187, 210, 234, 236, 268 |
| src/xngin/stats/power.py                                             |       65 |        5 |     92% |45, 84, 141-151 |
| src/xngin/xsecrets/gcp\_kms\_provider.py                             |       71 |       26 |     63% |62-77, 84-85, 102-103, 106, 110-118, 122-129 |
| src/xngin/xsecrets/provider.py                                       |       19 |        1 |     95% |        46 |
| src/xngin/xsecrets/secretservice.py                                  |       61 |        4 |     93% |42-43, 101, 123 |
| src/xngin/xsecrets/test\_gcp\_kms\_provider.py                       |      103 |       26 |     75% |40-42, 170-175, 182-189, 195-199, 206, 213-224 |
| src/xngin/xsecrets/test\_nacl\_provider.py                           |       67 |        1 |     99% |        24 |
|                                                            **TOTAL** | **7632** | **1067** | **86%** |           |

54 files skipped due to complete coverage.


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