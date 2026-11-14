# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/agency-fund/evidential-be/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                               |    Stmts |     Miss |   Cover |   Missing |
|----------------------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/xngin/apiserver/apikeys.py                                                     |       51 |        4 |     92% |     59-63 |
| src/xngin/apiserver/benchmarks/test\_draws\_perf.py                                |       91 |       62 |     32% |48, 52-93, 111-128, 143-159, 163, 171, 177-180, 192-215, 228-251, 264-278 |
| src/xngin/apiserver/certs/certs.py                                                 |       16 |        9 |     44% |19-23, 37-44 |
| src/xngin/apiserver/common\_field\_types.py                                        |       14 |        1 |     93% |        23 |
| src/xngin/apiserver/customlogging.py                                               |      105 |       14 |     87% |26-27, 145, 161-162, 169, 207-216, 224-225 |
| src/xngin/apiserver/database.py                                                    |       53 |        4 |     92% |40, 63, 69, 76 |
| src/xngin/apiserver/dependencies.py                                                |       12 |        1 |     92% |        12 |
| src/xngin/apiserver/dns/safe\_resolve.py                                           |       68 |       14 |     79% |47-58, 90, 113-114 |
| src/xngin/apiserver/dns/test\_safe\_resolve.py                                     |       45 |        2 |     96% |    49, 53 |
| src/xngin/apiserver/dwh/dwh\_errors.py                                             |      258 |       11 |     96% |158, 287, 309, 390-392, 400-402, 428, 434 |
| src/xngin/apiserver/dwh/dwh\_session.py                                            |      268 |       26 |     90% |77-79, 173-174, 243, 253, 296, 298, 308, 310, 333-334, 375, 388-397, 677-684, 688, 743, 759 |
| src/xngin/apiserver/dwh/dwh\_utils.py                                              |       37 |        3 |     92% |50, 57, 83 |
| src/xngin/apiserver/dwh/field\_aliases.py                                          |       27 |        1 |     96% |        42 |
| src/xngin/apiserver/dwh/google\_sheets.py                                          |      265 |        9 |     97% |51, 185, 263-265, 284-292, 295, 400 |
| src/xngin/apiserver/dwh/inspection\_types.py                                       |       55 |        6 |     89% |27, 45, 59, 68, 79, 85 |
| src/xngin/apiserver/dwh/inspections.py                                             |       19 |        1 |     95% |        33 |
| src/xngin/apiserver/dwh/participant\_metrics\_queries.py                           |      140 |        5 |     96% |87, 175, 269, 271, 298 |
| src/xngin/apiserver/dwh/queries.py                                                 |       47 |        3 |     94% |92, 95, 122 |
| src/xngin/apiserver/dwh/query\_constructors.py                                     |       80 |        4 |     95% |70-71, 90-91 |
| src/xngin/apiserver/dwh/test\_dialect\_sql.py                                      |       74 |        6 |     92% |494, 507, 510-513 |
| src/xngin/apiserver/dwh/test\_dwh\_errors\_bigquery.py                             |      259 |      170 |     34% |67, 72-74, 80-96, 102, 112-126, 131-137, 143-146, 152-156, 165-171, 178-189, 195-207, 215-249, 257-268, 274-286, 293-316, 324-326, 333-335, 342-349, 358-359, 366-382, 400-430 |
| src/xngin/apiserver/dwh/test\_dwh\_errors\_postgres.py                             |      223 |        1 |     99% |        49 |
| src/xngin/apiserver/dwh/test\_dwh\_identifiers\_bigquery.py                        |       77 |       44 |     43% |49-51, 58-59, 67-68, 82-115, 126-127, 131-132, 136-137, 143-144, 149-151, 167-171 |
| src/xngin/apiserver/dwh/test\_dwh\_identifiers\_redshift.py                        |       34 |       10 |     71% |47-48, 53-54, 60, 66, 70, 76, 81-82 |
| src/xngin/apiserver/dwh/test\_dwh\_records\_bigquery.py                            |       28 |       13 |     54% |24-25, 32-56 |
| src/xngin/apiserver/dwh/test\_dwh\_session.py                                      |      194 |        1 |     99% |       131 |
| src/xngin/apiserver/dwh/test\_dwh\_time\_types\_bigquery.py                        |       16 |        3 |     81% | 22-23, 30 |
| src/xngin/apiserver/dwh/test\_dwh\_time\_types\_redshift.py                        |       21 |        4 |     81% |39-40, 46, 52 |
| src/xngin/apiserver/dwh/test\_google\_sheets.py                                    |      834 |        6 |     99% |98, 133, 156-159 |
| src/xngin/apiserver/dwh/test\_queries.py                                           |       71 |        1 |     99% |       170 |
| src/xngin/apiserver/dwh/test\_query\_constructors.py                               |      160 |        2 |     99% |  444, 461 |
| src/xngin/apiserver/exceptionhandlers.py                                           |       74 |        4 |     95% |51, 67, 78, 92 |
| src/xngin/apiserver/exceptions\_common.py                                          |       51 |        2 |     96% |   127-128 |
| src/xngin/apiserver/flags.py                                                       |       61 |        4 |     93% |101, 104, 122, 125 |
| src/xngin/apiserver/main.py                                                        |       31 |        2 |     94% |    40, 48 |
| src/xngin/apiserver/openapi.py                                                     |       27 |        3 |     89% |113, 178, 180 |
| src/xngin/apiserver/pagination.py                                                  |      116 |       14 |     88% |47-48, 53-55, 61, 68, 98, 233, 235, 246-249 |
| src/xngin/apiserver/request\_encapsulation\_middleware.py                          |       69 |        3 |     96% |   113-115 |
| src/xngin/apiserver/routers/admin/admin\_api.py                                    |      540 |       20 |     96% |254, 266, 277, 281, 505, 953-957, 1143, 1179, 1328, 1377-1383, 1398, 1405, 1433, 1675 |
| src/xngin/apiserver/routers/admin/admin\_api\_converters.py                        |       62 |        7 |     89% |75-76, 86, 114-115, 125-126 |
| src/xngin/apiserver/routers/admin/admin\_api\_types.py                             |      133 |        2 |     98% |    35, 37 |
| src/xngin/apiserver/routers/admin/generic\_handlers.py                             |       21 |        1 |     95% |        47 |
| src/xngin/apiserver/routers/admin/test\_admin\_api.py                              |     1832 |        3 |     99% |2826, 2839-2840 |
| src/xngin/apiserver/routers/admin/test\_admin\_extra.py                            |      111 |        5 |     95% |99, 130-131, 159-160 |
| src/xngin/apiserver/routers/admin/test\_admin\_users\_api.py                       |      378 |        1 |     99% |        34 |
| src/xngin/apiserver/routers/admin\_integrations/admin\_integrations\_api.py        |      152 |        2 |     99% |  174, 395 |
| src/xngin/apiserver/routers/admin\_integrations/admin\_integrations\_api\_types.py |       15 |        1 |     93% |        26 |
| src/xngin/apiserver/routers/auth/auth\_api.py                                      |      167 |        1 |     99% |       201 |
| src/xngin/apiserver/routers/auth/auth\_dependencies.py                             |      105 |        3 |     97% |211, 233-235 |
| src/xngin/apiserver/routers/auth/discovery.py                                      |      218 |        6 |     97% |142, 177, 203, 324, 327, 330 |
| src/xngin/apiserver/routers/auth/oidc\_settings.py                                 |       70 |        3 |     96% |   143-145 |
| src/xngin/apiserver/routers/auth/test\_auth\_api.py                                |      450 |        1 |     99% |       165 |
| src/xngin/apiserver/routers/auth/test\_auth\_dependencies.py                       |      177 |        7 |     96% | 51, 58-64 |
| src/xngin/apiserver/routers/auth/test\_discovery.py                                |      233 |        1 |     99% |        90 |
| src/xngin/apiserver/routers/auth/token\_cryptor.py                                 |       43 |        4 |     91% |16-17, 53-54 |
| src/xngin/apiserver/routers/common\_api\_types.py                                  |      451 |       27 |     94% |225, 227, 579, 581, 583, 640, 956, 959, 978, 1270, 1279, 1282-1283, 1293, 1295, 1305, 1307, 1325, 1628, 1630, 1645, 1652, 1842, 2027, 2029-2031 |
| src/xngin/apiserver/routers/common\_enums.py                                       |      183 |       18 |     90% |67, 69, 95, 97, 99, 106, 161-162, 221, 276-279, 288, 327-328, 332, 364 |
| src/xngin/apiserver/routers/experiments/experiments\_api.py                        |       97 |        4 |     96% |132-134, 356 |
| src/xngin/apiserver/routers/experiments/experiments\_common.py                     |      504 |       26 |     95% |278, 398-399, 422, 500, 511, 553-554, 565, 600, 605, 710, 805-806, 831, 963, 967, 988-989, 992, 1121, 1191-1192, 1229, 1240, 1366 |
| src/xngin/apiserver/routers/experiments/experiments\_common\_csv.py                |       89 |        4 |     96% |42, 105, 239-240 |
| src/xngin/apiserver/routers/experiments/experiments\_dependencies.py               |       46 |        3 |     93% |54, 75, 82 |
| src/xngin/apiserver/routers/experiments/property\_filters.py                       |       94 |        5 |     95% |94-95, 145, 157-158 |
| src/xngin/apiserver/routers/experiments/test\_experiments\_api.py                  |      587 |        7 |     99% |76, 190-191, 1212, 1217-1218, 1257 |
| src/xngin/apiserver/routers/experiments/test\_experiments\_common.py               |     1321 |       11 |     99% |267-268, 278, 1702-1704, 2289-2290, 2738, 2961-2962 |
| src/xngin/apiserver/routers/experiments/test\_property\_filters.py                 |       58 |        1 |     98% |        25 |
| src/xngin/apiserver/routers/healthchecks\_api.py                                   |       32 |        1 |     97% |        55 |
| src/xngin/apiserver/routers/test\_assignment\_adapters.py                          |      235 |        1 |     99% |       103 |
| src/xngin/apiserver/settings.py                                                    |      132 |       21 |     84% |65, 72, 123-124, 184, 189, 195-196, 248-251, 270, 292, 303, 305, 330, 339, 342, 356, 359 |
| src/xngin/apiserver/sql/queries.py                                                 |       33 |        4 |     88% | 21, 56-58 |
| src/xngin/apiserver/sqla/tables.py                                                 |      339 |        3 |     99% |57, 415, 419 |
| src/xngin/apiserver/storage/bootstrap.py                                           |       93 |        6 |     94% |88, 307, 310, 412-414 |
| src/xngin/apiserver/storage/bootstrap\_snapshots.py                                |      157 |       42 |     73% |71, 81-89, 106, 121, 123, 126-127, 133, 136, 138, 143, 148, 153, 243-245, 326, 347-352, 368-395 |
| src/xngin/apiserver/storage/storage\_format\_converters.py                         |      208 |       11 |     95% |49, 150, 155-156, 273, 311, 329, 366, 502, 563-564 |
| src/xngin/apiserver/storage/test\_bootstrap\_bigquery.py                           |       21 |        6 |     71% |26-27, 38-43 |
| src/xngin/apiserver/test\_handler\_routes.py                                       |       49 |        4 |     92% |90, 93, 99, 103 |
| src/xngin/apiserver/test\_request\_context\_middleware.py                          |      148 |        2 |     99% |    45, 49 |
| src/xngin/apiserver/testing/assertions.py                                          |       68 |        6 |     91% |20, 41, 58, 61, 69, 103 |
| src/xngin/apiserver/testing/database\_plugin.py                                    |       33 |       11 |     67% |34-37, 39-42, 54-57 |
| src/xngin/apiserver/testing/helpers.py                                             |      104 |       21 |     80% |39, 58, 73, 75, 88, 98, 102, 104, 108, 148, 165-179 |
| src/xngin/apiserver/testing/stress\_dwh\_def.py                                    |       73 |        1 |     99% |       299 |
| src/xngin/apiserver/testing/stress\_loaders.py                                     |       56 |       16 |     71% |67-79, 85-106 |
| src/xngin/apiserver/testing/worker\_helpers.py                                     |      167 |       13 |     92% |125, 188, 236-238, 244, 247, 263, 269, 277-279, 295 |
| src/xngin/apiserver/worker/attempts.py                                             |      123 |        4 |     97% |242-243, 257-258 |
| src/xngin/apiserver/worker/jobs/bandit\_outcomes.py                                |       34 |        1 |     97% |        68 |
| src/xngin/apiserver/worker/jobs/base.py                                            |       39 |        1 |     97% |        62 |
| src/xngin/apiserver/worker/jobs/snapshot\_analysis.py                              |       58 |        2 |     97% |  102, 113 |
| src/xngin/apiserver/worker/jobs/test\_autofail.py                                  |       68 |        2 |     97% |     69-70 |
| src/xngin/apiserver/worker/jobs/test\_snapshot\_analysis.py                        |      214 |        9 |     96% |57-63, 426-427 |
| src/xngin/apiserver/worker/scheduler.py                                            |       81 |        2 |     98% |   103-104 |
| src/xngin/apiserver/worker/tq/handlers.py                                          |       85 |       11 |     87% |82-83, 129, 137-138, 149-158, 187-188 |
| src/xngin/apiserver/worker/tq/task\_queue\_thread.py                               |       30 |        1 |     97% |        57 |
| src/xngin/apiserver/worker/tq/test\_tq.py                                          |      175 |        2 |     99% |  103, 118 |
| src/xngin/apiserver/worker/tq/tq\_test\_support.py                                 |       48 |        5 |     90% |27-28, 46, 48, 70 |
| src/xngin/cli/commands/create\_testing\_dwh.py                                     |      254 |       75 |     70% |76, 95, 106, 161, 212, 233-243, 247-252, 269-270, 274-275, 279-314, 319-327, 332, 341-376, 528, 531-532, 559, 567, 573 |
| src/xngin/cli/commands/databases.py                                                |       86 |       60 |     30% |40-52, 57-61, 66-83, 92-115, 130-137, 142-152, 167-178 |
| src/xngin/cli/commands/test\_create\_testing\_dwh\_bigquery.py                     |       34 |       17 |     50% |27-36, 42-58 |
| src/xngin/cli/common.py                                                            |       42 |        6 |     86% |34-35, 40-41, 64, 67 |
| src/xngin/cli/main.py                                                              |      188 |      130 |     31% |40-45, 51-59, 103-110, 125-138, 151-161, 165-168, 226-277, 301-312, 323-324, 332-333, 340-402, 425-463, 467 |
| src/xngin/db\_extensions/custom\_functions.py                                      |       29 |        2 |     93% |    35, 55 |
| src/xngin/events/common.py                                                         |       12 |        1 |     92% |        20 |
| src/xngin/events/experiment\_created.py                                            |       13 |        1 |     92% |        24 |
| src/xngin/ops/sentry.py                                                            |       17 |       10 |     41% |     18-54 |
| src/xngin/stats/assignment.py                                                      |       92 |        2 |     98% |  186, 280 |
| src/xngin/stats/balance.py                                                         |       78 |        3 |     96% |110, 141, 210 |
| src/xngin/stats/bandit\_analysis.py                                                |       73 |        4 |     95% |134, 136, 199-200 |
| src/xngin/stats/bandit\_sampling.py                                                |       86 |        7 |     92% |184, 219, 226, 254, 282, 284, 316 |
| src/xngin/stats/bandit\_weights\_to\_prior.py                                      |       48 |        3 |     94% |30, 78, 122 |
| src/xngin/stats/cluster\_icc.py                                                    |       71 |        2 |     97% |    38, 62 |
| src/xngin/stats/cluster\_power.py                                                  |      123 |        4 |     97% |201, 257, 260, 354 |
| src/xngin/stats/individual\_power.py                                               |      162 |        7 |     96% |110, 113, 160-161, 225, 329, 446 |
| src/xngin/stats/power.py                                                           |       60 |        1 |     98% |       257 |
| src/xngin/stats/stats\_errors.py                                                   |       28 |        3 |     89% |10, 37, 45 |
| src/xngin/xsecrets/chafernet.py                                                    |       52 |        1 |     98% |        92 |
| src/xngin/xsecrets/gcp\_kms\_provider.py                                           |       70 |       28 |     60% |64-79, 86-87, 104-108, 111, 115-123, 127-134 |
| src/xngin/xsecrets/provider.py                                                     |       19 |        1 |     95% |        46 |
| src/xngin/xsecrets/secretservice.py                                                |       64 |        7 |     89% |37, 45-46, 51-52, 104, 126 |
| src/xngin/xsecrets/test\_gcp\_kms\_provider.py                                     |      103 |       26 |     75% |40-42, 170-175, 182-189, 195-199, 206, 213-224 |
| src/xngin/xsecrets/test\_nacl\_provider.py                                         |       67 |        1 |     99% |        24 |
| **TOTAL**                                                                          | **23325** | **1258** | **95%** |           |

126 files skipped due to complete coverage.


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