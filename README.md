# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/agency-fund/evidential-be/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                 |    Stmts |     Miss |   Cover |   Missing |
|--------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/xngin/apiserver/apikeys.py                                       |       39 |        8 |     79% |35, 37-41, 72-75 |
| src/xngin/apiserver/certs/certs.py                                   |       16 |        9 |     44% |21-25, 39-46 |
| src/xngin/apiserver/common\_field\_types.py                          |       12 |        1 |     92% |        13 |
| src/xngin/apiserver/conftest.py                                      |      178 |        8 |     96% |72, 94-96, 106, 145, 147, 151 |
| src/xngin/apiserver/customlogging.py                                 |       54 |        5 |     91% |41-42, 47, 62-63 |
| src/xngin/apiserver/database.py                                      |       61 |       16 |     74% |28, 42, 59, 65, 72, 93-109 |
| src/xngin/apiserver/dependencies.py                                  |       27 |        9 |     67% |22, 49, 52-56, 61-63 |
| src/xngin/apiserver/dns/safe\_resolve.py                             |       44 |        9 |     80% |28-29, 33-34, 47, 74, 77, 82-83 |
| src/xngin/apiserver/dwh/dwh\_session.py                              |      179 |       77 |     57% |37, 43, 71-76, 79, 142, 150, 167-178, 185-246, 272-274, 289, 342-349, 352, 356-362, 380, 414-421, 429 |
| src/xngin/apiserver/dwh/inspection\_types.py                         |       61 |        5 |     92% |41, 61, 85, 96, 104 |
| src/xngin/apiserver/dwh/inspections.py                               |       28 |        1 |     96% |        65 |
| src/xngin/apiserver/dwh/queries.py                                   |      171 |       33 |     81% |124-167, 183, 195, 244, 282-284, 336, 357, 376-377, 386, 388, 391-392, 407-408 |
| src/xngin/apiserver/dwh/test\_dialect\_sql.py                        |       77 |        6 |     92% |535, 549, 552-557 |
| src/xngin/apiserver/dwh/test\_queries.py                             |      224 |        1 |     99% |       216 |
| src/xngin/apiserver/exceptionhandlers.py                             |       45 |        9 |     80% |30, 36, 46, 56-61, 83 |
| src/xngin/apiserver/main.py                                          |       30 |        3 |     90% | 19-21, 47 |
| src/xngin/apiserver/openapi.py                                       |       20 |        9 |     55% |     19-68 |
| src/xngin/apiserver/routers/admin/admin\_api.py                      |      501 |      254 |     49% |162, 235-240, 267-272, 297-302, 316-320, 333-339, 362, 384, 395, 410-423, 443-453, 485-502, 520-537, 572-584, 588-601, 623-643, 660-679, 694-700, 717-727, 759-770, 798-824, 835-853, 863-867, 886-893, 899-907, 929-943, 957, 1011-1021, 1036-1123, 1135, 1150-1175, 1202-1208, 1262-1267, 1322-1331, 1356-1389, 1448, 1475-1476, 1490-1491, 1502, 1514-1522, 1533-1539, 1556-1563, 1595-1606, 1647-1673, 1686-1690 |
| src/xngin/apiserver/routers/admin/admin\_api\_types.py               |      144 |        2 |     99% |    31, 33 |
| src/xngin/apiserver/routers/admin/generic\_handlers.py               |       24 |       14 |     42% |     41-59 |
| src/xngin/apiserver/routers/assignment\_adapters.py                  |       44 |        1 |     98% |       159 |
| src/xngin/apiserver/routers/auth/auth\_api.py                        |       35 |       12 |     66% |29-34, 65-88 |
| src/xngin/apiserver/routers/auth/auth\_dependencies.py               |       96 |       47 |     51% |56, 71-79, 86-117, 137, 144-187, 218-220 |
| src/xngin/apiserver/routers/common\_api\_types.py                    |      335 |       46 |     86% |106, 135, 137, 240, 310-319, 339, 550-554, 595, 766-789, 796-806, 813-819, 1016 |
| src/xngin/apiserver/routers/common\_enums.py                         |      167 |       55 |     67% |73, 75, 87, 89, 98-107, 112, 134-151, 162-166, 207, 245-258, 278-284, 312-316 |
| src/xngin/apiserver/routers/experiments/dependencies.py              |       18 |        3 |     83% |     51-57 |
| src/xngin/apiserver/routers/experiments/experiments\_api.py          |       63 |       15 |     76% |97, 106-119, 141, 154, 181, 192, 206, 230-235, 252-259 |
| src/xngin/apiserver/routers/experiments/experiments\_common.py       |      296 |       87 |     71% |64, 69, 94, 110, 118, 143-153, 173, 195, 211-220, 244, 255, 335, 349, 374-392, 424-442, 489-496, 552-570, 624-626, 703, 707, 730, 753, 766, 794-882 |
| src/xngin/apiserver/routers/experiments/test\_experiments\_common.py |      343 |        1 |     99% |       965 |
| src/xngin/apiserver/routers/healthchecks\_api.py                     |       16 |        2 |     88% |     26-29 |
| src/xngin/apiserver/routers/test\_common\_api\_types.py              |       41 |        1 |     98% |        76 |
| src/xngin/apiserver/settings.py                                      |      179 |       34 |     81% |105, 127, 134, 140, 200, 202-208, 270, 277, 280-285, 339-345, 366, 390, 400-405, 415, 418, 435, 457, 472 |
| src/xngin/apiserver/sqla/tables.py                                   |      262 |       12 |     95% |57, 170-172, 325, 347, 371, 376-378, 381-382 |
| src/xngin/apiserver/storage/storage\_format\_converters.py           |       90 |       16 |     82% |59, 76, 90-91, 112, 165-197, 291, 325-364 |
| src/xngin/apiserver/testing/assertions.py                            |       14 |        5 |     64% | 16-19, 27 |
| src/xngin/apiserver/testing/xurl.py                                  |       29 |        1 |     97% |        37 |
| src/xngin/cli/main.py                                                |      336 |      197 |     41% |85-101, 117-149, 153-157, 169-171, 266, 269, 274, 286-287, 296, 304-322, 325-361, 363-366, 386-391, 396-397, 428-433, 446-454, 460-466, 472-475, 484-492, 517-522, 537-545, 558-566, 570-573, 615-702, 720-724, 737-738, 748-750, 754 |
| src/xngin/db\_extensions/custom\_functions.py                        |       41 |        3 |     93% |47, 71, 84 |
| src/xngin/db\_extensions/test\_custom\_functions.py                  |       48 |       12 |     75% |59-68, 73-82 |
| src/xngin/events/common.py                                           |        7 |        2 |     71% |    14, 21 |
| src/xngin/events/experiment\_created.py                              |       14 |        4 |     71% | 18, 21-25 |
| src/xngin/events/webhook\_sent.py                                    |       15 |        5 |     67% |     18-23 |
| src/xngin/stats/balance.py                                           |       62 |        3 |     95% |113, 143, 146 |
| src/xngin/stats/bandit\_sampling.py                                  |       70 |       59 |     16% |27-29, 50-57, 76-78, 100-115, 141-164, 183-215, 237-277 |
| src/xngin/stats/power.py                                             |       65 |        5 |     92% |51, 94, 151-161 |
| src/xngin/xsecrets/gcp\_kms\_provider.py                             |       71 |       26 |     63% |62-79, 86-89, 106-107, 110, 114-122, 126-133 |
| src/xngin/xsecrets/provider.py                                       |       19 |        1 |     95% |        46 |
| src/xngin/xsecrets/secretservice.py                                  |       61 |        4 |     93% |42-43, 106, 130 |
| src/xngin/xsecrets/test\_gcp\_kms\_provider.py                       |      103 |       26 |     75% |40-42, 170-175, 182-189, 195-203, 210, 217-228 |
| src/xngin/xsecrets/test\_nacl\_provider.py                           |       67 |        1 |     99% |        24 |
|                                                            **TOTAL** | **6674** | **1165** | **83%** |           |

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