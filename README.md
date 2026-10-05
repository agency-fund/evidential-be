# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/agency-fund/evidential-be/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                               |    Stmts |     Miss |   Cover |   Missing |
|----------------------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/xngin/apiserver/apikeys.py                                                     |       51 |        4 |     92% |     59-63 |
| src/xngin/apiserver/benchmarks/test\_draws\_perf.py                                |       95 |       66 |     31% |48, 52-93, 111-128, 143-159, 163, 171, 177-180, 192-215, 228-251, 264-282 |
| src/xngin/apiserver/certs/certs.py                                                 |       16 |        9 |     44% |19-23, 37-44 |
| src/xngin/apiserver/common\_field\_types.py                                        |       12 |        1 |     92% |        13 |
| src/xngin/apiserver/conftest.py                                                    |      243 |       25 |     90% |71, 90, 105, 107, 118, 161, 165, 167, 171, 394, 411-425, 440, 457, 460, 495 |
| src/xngin/apiserver/customlogging.py                                               |       66 |       13 |     80% |25-26, 48-68, 73-74, 102-107 |
| src/xngin/apiserver/database.py                                                    |       46 |        5 |     89% |29, 40, 57, 63, 70 |
| src/xngin/apiserver/dependencies.py                                                |       12 |        1 |     92% |        12 |
| src/xngin/apiserver/dns/safe\_resolve.py                                           |       68 |       14 |     79% |47-58, 90, 113-114 |
| src/xngin/apiserver/dns/test\_safe\_resolve.py                                     |       45 |        2 |     96% |    49, 53 |
| src/xngin/apiserver/dwh/dwh\_session.py                                            |      211 |       54 |     74% |74, 158-159, 199, 201-202, 213-268, 293-295, 310, 449-456, 460, 467-469, 488, 490-491, 502 |
| src/xngin/apiserver/dwh/dwh\_utils.py                                              |       17 |        3 |     82% |20, 27, 35 |
| src/xngin/apiserver/dwh/inspection\_types.py                                       |       55 |        5 |     91% |27, 45, 68, 79, 85 |
| src/xngin/apiserver/dwh/inspections.py                                             |       33 |        2 |     94% |   73, 101 |
| src/xngin/apiserver/dwh/participant\_metrics\_queries.py                           |      144 |        7 |     95% |73-79, 166, 263, 265, 295 |
| src/xngin/apiserver/dwh/queries.py                                                 |       45 |        3 |     93% |83, 86, 112 |
| src/xngin/apiserver/dwh/query\_constructors.py                                     |       81 |        4 |     95% |75-76, 95-96 |
| src/xngin/apiserver/dwh/test\_dialect\_sql.py                                      |       74 |        6 |     92% |494, 507, 510-513 |
| src/xngin/apiserver/dwh/test\_dwh\_session.py                                      |       97 |        1 |     99% |       121 |
| src/xngin/apiserver/dwh/test\_queries.py                                           |       69 |        1 |     99% |       170 |
| src/xngin/apiserver/dwh/test\_query\_constructors.py                               |      160 |        2 |     99% |  444, 461 |
| src/xngin/apiserver/dwhpull/cli.py                                                 |       25 |        7 |     72% |     51-59 |
| src/xngin/apiserver/dwhpull/dwhpull.py                                             |       87 |        1 |     99% |       115 |
| src/xngin/apiserver/exceptionhandlers.py                                           |       77 |        8 |     90% |56, 72-77, 91, 116 |
| src/xngin/apiserver/flags.py                                                       |       61 |        5 |     92% |50, 99, 102, 120, 123 |
| src/xngin/apiserver/main.py                                                        |       38 |        6 |     84% |40, 48, 77-78, 95-97 |
| src/xngin/apiserver/openapi.py                                                     |       27 |        3 |     89% |113, 178, 180 |
| src/xngin/apiserver/pagination.py                                                  |      116 |       14 |     88% |47-48, 53-55, 61, 68, 98, 233, 235, 246-249 |
| src/xngin/apiserver/request\_encapsulation\_middleware.py                          |       69 |        3 |     96% |   113-115 |
| src/xngin/apiserver/routers/admin/admin\_api.py                                    |      545 |       21 |     96% |270, 282, 293, 297, 521, 992-996, 1163, 1199, 1347, 1395-1401, 1416, 1423, 1451, 1641, 1697 |
| src/xngin/apiserver/routers/admin/admin\_api\_converters.py                        |       62 |        8 |     87% |29, 73-74, 84, 110-111, 123-124 |
| src/xngin/apiserver/routers/admin/admin\_api\_types.py                             |      125 |        2 |     98% |    34, 36 |
| src/xngin/apiserver/routers/admin/generic\_handlers.py                             |       21 |        1 |     95% |        47 |
| src/xngin/apiserver/routers/admin/test\_admin\_api.py                              |     1775 |        3 |     99% |2710, 2723-2724 |
| src/xngin/apiserver/routers/admin/test\_admin\_extra.py                            |      111 |        5 |     95% |98, 129-130, 158-159 |
| src/xngin/apiserver/routers/admin/test\_admin\_users\_api.py                       |      378 |        1 |     99% |        34 |
| src/xngin/apiserver/routers/admin\_integrations/admin\_integrations\_api.py        |      152 |        2 |     99% |  174, 395 |
| src/xngin/apiserver/routers/admin\_integrations/admin\_integrations\_api\_types.py |       15 |        1 |     93% |        26 |
| src/xngin/apiserver/routers/auth/auth\_api.py                                      |      167 |        1 |     99% |       201 |
| src/xngin/apiserver/routers/auth/auth\_dependencies.py                             |      106 |        3 |     97% |205, 227-229 |
| src/xngin/apiserver/routers/auth/discovery.py                                      |      218 |        6 |     97% |142, 177, 203, 324, 327, 330 |
| src/xngin/apiserver/routers/auth/oidc\_settings.py                                 |       70 |        3 |     96% |   143-145 |
| src/xngin/apiserver/routers/auth/test\_auth\_api.py                                |      450 |        1 |     99% |       165 |
| src/xngin/apiserver/routers/auth/test\_auth\_dependencies.py                       |      172 |        7 |     96% | 50, 57-63 |
| src/xngin/apiserver/routers/auth/test\_discovery.py                                |      233 |        1 |     99% |        90 |
| src/xngin/apiserver/routers/auth/token\_cryptor.py                                 |       43 |        4 |     91% |16-17, 53-54 |
| src/xngin/apiserver/routers/common\_api\_types.py                                  |      395 |       30 |     92% |188, 190, 484, 486, 488, 545, 821-828, 847, 1126, 1135, 1138-1139, 1149, 1151, 1161, 1163, 1181, 1463, 1465, 1475, 1482, 1672, 1853, 1855-1857 |
| src/xngin/apiserver/routers/common\_enums.py                                       |      184 |       19 |     90% |67, 69, 93, 95, 97, 99, 106, 161-162, 214, 269-272, 281, 320-321, 325, 357 |
| src/xngin/apiserver/routers/experiments/experiments\_api.py                        |       97 |        4 |     96% |132-134, 356 |
| src/xngin/apiserver/routers/experiments/experiments\_common.py                     |      479 |       23 |     95% |372-373, 396, 474, 485, 527-528, 539, 558, 649, 744-745, 770, 902, 906, 928-929, 932, 1061, 1145-1146, 1183, 1319 |
| src/xngin/apiserver/routers/experiments/experiments\_common\_csv.py                |       89 |        4 |     96% |43, 106, 240-241 |
| src/xngin/apiserver/routers/experiments/experiments\_dependencies.py               |       46 |        3 |     93% |54, 75, 82 |
| src/xngin/apiserver/routers/experiments/property\_filters.py                       |       96 |        8 |     92% |25, 28, 32, 95-96, 148, 160-161 |
| src/xngin/apiserver/routers/experiments/test\_experiments\_api.py                  |      587 |        7 |     99% |76, 190-191, 1212, 1217-1218, 1257 |
| src/xngin/apiserver/routers/experiments/test\_experiments\_common.py               |     1197 |        9 |     99% |260-261, 271, 1580-1582, 2057-2058, 2506 |
| src/xngin/apiserver/routers/experiments/test\_property\_filters.py                 |       41 |        1 |     98% |        24 |
| src/xngin/apiserver/routers/healthchecks\_api.py                                   |       16 |        2 |     88% |     26-27 |
| src/xngin/apiserver/routers/power\_adapters.py                                     |       33 |        1 |     97% |        83 |
| src/xngin/apiserver/routers/test\_assignment\_adapters.py                          |      235 |        1 |     99% |       103 |
| src/xngin/apiserver/settings.py                                                    |      119 |       21 |     82% |63, 70, 76, 121-122, 182, 187, 193-194, 246-249, 268, 290, 301, 303, 313, 316, 341, 344 |
| src/xngin/apiserver/snapshots/autofail.py                                          |       73 |        2 |     97% |   127-128 |
| src/xngin/apiserver/snapshots/cli.py                                               |       39 |       15 |     62% |30-37, 111-121 |
| src/xngin/apiserver/snapshots/fake\_data.py                                        |      125 |       37 |     70% |72-80, 86, 89, 91, 96, 101, 106, 198-201, 279, 300-305, 321-348 |
| src/xngin/apiserver/snapshots/snapshotter.py                                       |       77 |        2 |     97% |  201, 212 |
| src/xngin/apiserver/snapshots/test\_autofail.py                                    |      197 |        2 |     99% |   101-102 |
| src/xngin/apiserver/snapshots/test\_snapshotter.py                                 |      290 |        8 |     97% |62-67, 683-684 |
| src/xngin/apiserver/sql/queries.py                                                 |       39 |       10 |     74% | 21, 58-67 |
| src/xngin/apiserver/sqla/tables.py                                                 |      336 |        4 |     99% |57, 222, 407, 411 |
| src/xngin/apiserver/storage/bootstrap.py                                           |       40 |        1 |     98% |        58 |
| src/xngin/apiserver/storage/storage\_format\_converters.py                         |      198 |       11 |     94% |48, 150, 155-156, 263, 301, 319, 356, 480, 541-542 |
| src/xngin/apiserver/test\_handler\_routes.py                                       |       49 |        4 |     92% |90, 93, 99, 103 |
| src/xngin/apiserver/testing/assertions.py                                          |        7 |        1 |     86% |         7 |
| src/xngin/cli/commands/create\_testing\_dwh.py                                     |      186 |      147 |     21% |30-32, 36-40, 51-82, 108, 119-129, 141-143, 148-155, 159-162, 166-170, 174-180, 187-207, 212-230, 234-271, 275-280, 284-308, 393-415 |
| src/xngin/cli/commands/databases.py                                                |       75 |       50 |     33% |40-52, 57-61, 66-83, 92-115, 130-137, 145-154 |
| src/xngin/cli/common.py                                                            |       38 |       13 |     66% |33-34, 39-40, 52-53, 61-63, 68-73 |
| src/xngin/cli/main.py                                                              |      188 |      130 |     31% |40-45, 51-59, 103-110, 125-138, 151-161, 165-168, 209-254, 278-289, 300-301, 309-310, 317-379, 402-440, 444 |
| src/xngin/db\_extensions/custom\_functions.py                                      |       29 |        2 |     93% |    35, 55 |
| src/xngin/db\_extensions/test\_custom\_functions.py                                |       41 |        6 |     85% |     58-67 |
| src/xngin/events/common.py                                                         |       12 |        1 |     92% |        20 |
| src/xngin/events/experiment\_created.py                                            |       13 |        1 |     92% |        24 |
| src/xngin/ops/sentry.py                                                            |       13 |        6 |     54% |     18-40 |
| src/xngin/stats/assignment.py                                                      |       87 |        2 |     98% |  170, 262 |
| src/xngin/stats/balance.py                                                         |       78 |        3 |     96% |110, 141, 210 |
| src/xngin/stats/bandit\_analysis.py                                                |       73 |        4 |     95% |134, 136, 199-200 |
| src/xngin/stats/bandit\_sampling.py                                                |       86 |        7 |     92% |184, 219, 226, 254, 282, 284, 316 |
| src/xngin/stats/bandit\_weights\_to\_prior.py                                      |       48 |        3 |     94% |30, 78, 122 |
| src/xngin/stats/cluster\_icc.py                                                    |       71 |        2 |     97% |    38, 62 |
| src/xngin/stats/cluster\_power.py                                                  |      114 |        2 |     98% |  251, 254 |
| src/xngin/stats/individual\_power.py                                               |      103 |        5 |     95% |77, 80, 124-125, 200 |
| src/xngin/stats/power.py                                                           |       59 |        1 |     98% |       245 |
| src/xngin/stats/stats\_errors.py                                                   |       25 |        3 |     88% |10, 37, 45 |
| src/xngin/tq/handlers.py                                                           |       85 |       11 |     87% |82-83, 129, 137-138, 149-158, 187-188 |
| src/xngin/tq/task\_queue.py                                                        |      100 |        2 |     98% |   241-242 |
| src/xngin/tq/tq\_test\_support.py                                                  |       48 |        5 |     90% |27-28, 46, 48, 70 |
| src/xngin/xsecrets/chafernet.py                                                    |       52 |        1 |     98% |        92 |
| src/xngin/xsecrets/gcp\_kms\_provider.py                                           |       70 |       28 |     60% |64-79, 86-87, 104-108, 111, 115-123, 127-134 |
| src/xngin/xsecrets/provider.py                                                     |       19 |        1 |     95% |        46 |
| src/xngin/xsecrets/secretservice.py                                                |       64 |        7 |     89% |37, 45-46, 51-52, 104, 126 |
| src/xngin/xsecrets/test\_gcp\_kms\_provider.py                                     |      103 |       26 |     75% |40-42, 170-175, 182-189, 195-199, 206, 213-224 |
| src/xngin/xsecrets/test\_nacl\_provider.py                                         |       67 |        1 |     99% |        24 |
| **TOTAL**                                                                          | **17701** | **1045** | **94%** |           |

89 files skipped due to complete coverage.


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