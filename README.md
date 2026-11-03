# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/agency-fund/evidential-be/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                               |    Stmts |     Miss |   Cover |   Missing |
|----------------------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/xngin/apiserver/apikeys.py                                                     |       51 |        4 |     92% |     59-63 |
| src/xngin/apiserver/benchmarks/test\_draws\_perf.py                                |       91 |       62 |     32% |48, 52-93, 111-128, 143-159, 163, 171, 177-180, 192-215, 228-251, 264-278 |
| src/xngin/apiserver/certs/certs.py                                                 |       16 |        9 |     44% |19-23, 37-44 |
| src/xngin/apiserver/common\_field\_types.py                                        |       14 |        1 |     93% |        23 |
| src/xngin/apiserver/customlogging.py                                               |       71 |       12 |     83% |25-26, 77, 87-88, 95, 130-139 |
| src/xngin/apiserver/database.py                                                    |       53 |        4 |     92% |40, 63, 69, 76 |
| src/xngin/apiserver/dependencies.py                                                |       12 |        1 |     92% |        12 |
| src/xngin/apiserver/dns/safe\_resolve.py                                           |       68 |       14 |     79% |47-58, 90, 113-114 |
| src/xngin/apiserver/dns/test\_safe\_resolve.py                                     |       45 |        2 |     96% |    49, 53 |
| src/xngin/apiserver/dwh/dwh\_session.py                                            |      244 |       64 |     74% |80-81, 92-94, 178-179, 218, 226, 229, 231-232, 243-307, 588-595, 599, 606-608, 627, 629-630, 641 |
| src/xngin/apiserver/dwh/dwh\_utils.py                                              |       25 |        3 |     88% |28, 35, 61 |
| src/xngin/apiserver/dwh/field\_aliases.py                                          |       27 |        1 |     96% |        42 |
| src/xngin/apiserver/dwh/inspection\_types.py                                       |       55 |        6 |     89% |27, 45, 59, 68, 79, 85 |
| src/xngin/apiserver/dwh/inspections.py                                             |       18 |        1 |     94% |        33 |
| src/xngin/apiserver/dwh/participant\_metrics\_queries.py                           |      140 |        7 |     95% |81-87, 175, 269, 271, 298 |
| src/xngin/apiserver/dwh/queries.py                                                 |       47 |        3 |     94% |92, 95, 122 |
| src/xngin/apiserver/dwh/query\_constructors.py                                     |       81 |        4 |     95% |75-76, 95-96 |
| src/xngin/apiserver/dwh/test\_dialect\_sql.py                                      |       74 |        6 |     92% |494, 507, 510-513 |
| src/xngin/apiserver/dwh/test\_dwh\_identifiers\_bigquery.py                        |       72 |       41 |     43% |47-49, 56-57, 65-66, 80-113, 124-125, 129-130, 134-135, 141-142, 147-149, 165-166 |
| src/xngin/apiserver/dwh/test\_dwh\_identifiers\_redshift.py                        |       34 |       10 |     71% |47-48, 53-54, 60, 66, 70, 76, 81-82 |
| src/xngin/apiserver/dwh/test\_dwh\_records\_bigquery.py                            |       28 |       13 |     54% |24-25, 32-56 |
| src/xngin/apiserver/dwh/test\_dwh\_session.py                                      |      123 |        1 |     99% |       123 |
| src/xngin/apiserver/dwh/test\_dwh\_time\_types\_bigquery.py                        |       16 |        3 |     81% | 22-23, 30 |
| src/xngin/apiserver/dwh/test\_dwh\_time\_types\_redshift.py                        |       21 |        4 |     81% |39-40, 46, 52 |
| src/xngin/apiserver/dwh/test\_queries.py                                           |       69 |        1 |     99% |       170 |
| src/xngin/apiserver/dwh/test\_query\_constructors.py                               |      160 |        2 |     99% |  444, 461 |
| src/xngin/apiserver/exceptionhandlers.py                                           |       80 |        5 |     94% |56, 75, 86, 100, 125 |
| src/xngin/apiserver/flags.py                                                       |       57 |        4 |     93% |87, 90, 108, 111 |
| src/xngin/apiserver/main.py                                                        |       38 |        6 |     84% |41, 49, 78-79, 96-98 |
| src/xngin/apiserver/openapi.py                                                     |       27 |        3 |     89% |113, 178, 180 |
| src/xngin/apiserver/pagination.py                                                  |      116 |       14 |     88% |47-48, 53-55, 61, 68, 98, 233, 235, 246-249 |
| src/xngin/apiserver/request\_encapsulation\_middleware.py                          |       69 |        3 |     96% |   113-115 |
| src/xngin/apiserver/routers/admin/admin\_api.py                                    |      535 |       20 |     96% |269, 281, 292, 296, 520, 968-972, 1139, 1175, 1323, 1371-1377, 1392, 1399, 1427, 1669 |
| src/xngin/apiserver/routers/admin/admin\_api\_converters.py                        |       58 |        8 |     86% |29, 73-74, 84, 110-111, 121-122 |
| src/xngin/apiserver/routers/admin/admin\_api\_types.py                             |      124 |        2 |     98% |    34, 36 |
| src/xngin/apiserver/routers/admin/generic\_handlers.py                             |       21 |        1 |     95% |        47 |
| src/xngin/apiserver/routers/admin/test\_admin\_api.py                              |     1780 |        3 |     99% |2745, 2758-2759 |
| src/xngin/apiserver/routers/admin/test\_admin\_extra.py                            |      111 |        5 |     95% |98, 129-130, 158-159 |
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
| src/xngin/apiserver/routers/common\_api\_types.py                                  |      426 |       30 |     93% |206, 208, 534, 536, 538, 595, 907-914, 933, 1219, 1228, 1231-1232, 1242, 1244, 1254, 1256, 1274, 1571, 1573, 1588, 1595, 1785, 1966, 1968-1970 |
| src/xngin/apiserver/routers/common\_enums.py                                       |      183 |       18 |     90% |67, 69, 95, 97, 99, 106, 161-162, 221, 276-279, 288, 327-328, 332, 364 |
| src/xngin/apiserver/routers/experiments/experiments\_api.py                        |       97 |        4 |     96% |132-134, 356 |
| src/xngin/apiserver/routers/experiments/experiments\_common.py                     |      491 |       24 |     95% |393-394, 417, 495, 506, 548-549, 560, 579, 670, 765-766, 791, 923, 927, 948-949, 952, 1081, 1151-1152, 1189, 1200, 1326 |
| src/xngin/apiserver/routers/experiments/experiments\_common\_csv.py                |       89 |        4 |     96% |43, 106, 240-241 |
| src/xngin/apiserver/routers/experiments/experiments\_dependencies.py               |       46 |        3 |     93% |54, 75, 82 |
| src/xngin/apiserver/routers/experiments/property\_filters.py                       |       93 |        5 |     95% |93-94, 146, 158-159 |
| src/xngin/apiserver/routers/experiments/test\_experiments\_api.py                  |      587 |        7 |     99% |76, 190-191, 1212, 1217-1218, 1257 |
| src/xngin/apiserver/routers/experiments/test\_experiments\_common.py               |     1320 |       11 |     99% |267-268, 278, 1702-1704, 2289-2290, 2738, 2961-2962 |
| src/xngin/apiserver/routers/experiments/test\_property\_filters.py                 |       58 |        1 |     98% |        25 |
| src/xngin/apiserver/routers/healthchecks\_api.py                                   |       32 |        1 |     97% |        55 |
| src/xngin/apiserver/routers/power\_adapters.py                                     |       38 |        1 |     97% |        90 |
| src/xngin/apiserver/routers/test\_assignment\_adapters.py                          |      235 |        1 |     99% |       103 |
| src/xngin/apiserver/settings.py                                                    |      118 |       21 |     82% |63, 70, 76, 121-122, 182, 187, 193-194, 246-249, 268, 290, 301, 303, 313, 316, 330, 333 |
| src/xngin/apiserver/sql/queries.py                                                 |       33 |        4 |     88% | 21, 56-58 |
| src/xngin/apiserver/sqla/tables.py                                                 |      339 |        3 |     99% |57, 415, 419 |
| src/xngin/apiserver/storage/bootstrap.py                                           |       90 |        5 |     94% |88, 307, 310, 410-411 |
| src/xngin/apiserver/storage/bootstrap\_snapshots.py                                |      124 |       36 |     71% |72-80, 86, 89, 91, 96, 101, 106, 195-197, 275, 296-301, 317-344 |
| src/xngin/apiserver/storage/storage\_format\_converters.py                         |      199 |       11 |     94% |48, 149, 154-155, 272, 310, 328, 365, 489, 550-551 |
| src/xngin/apiserver/storage/test\_bootstrap\_bigquery.py                           |       21 |        6 |     71% |26-27, 38-43 |
| src/xngin/apiserver/test\_handler\_routes.py                                       |       49 |        4 |     92% |90, 93, 99, 103 |
| src/xngin/apiserver/test\_request\_context\_middleware.py                          |      143 |        2 |     99% |    40, 44 |
| src/xngin/apiserver/testing/assertions.py                                          |       61 |        5 |     92% |20, 41, 58, 61, 96 |
| src/xngin/apiserver/testing/database\_plugin.py                                    |       33 |       12 |     64% |34-37, 39-42, 48, 54-57 |
| src/xngin/apiserver/testing/helpers.py                                             |      104 |       21 |     80% |39, 58, 73, 75, 88, 98, 102, 104, 108, 148, 165-179 |
| src/xngin/apiserver/testing/pg\_helpers.py                                         |       90 |        8 |     91% |   149-157 |
| src/xngin/apiserver/testing/stress\_dwh\_def.py                                    |       73 |        1 |     99% |       299 |
| src/xngin/apiserver/testing/stress\_loaders.py                                     |       56 |       16 |     71% |67-79, 85-106 |
| src/xngin/apiserver/testing/worker\_helpers.py                                     |      167 |       13 |     92% |125, 188, 236-238, 244, 247, 263, 269, 277-279, 295 |
| src/xngin/apiserver/worker/attempts.py                                             |      120 |        4 |     97% |231-232, 246-247 |
| src/xngin/apiserver/worker/jobs/bandit\_outcomes.py                                |       34 |        1 |     97% |        68 |
| src/xngin/apiserver/worker/jobs/base.py                                            |       31 |        1 |     97% |        48 |
| src/xngin/apiserver/worker/jobs/snapshot\_analysis.py                              |       51 |        2 |     96% |   93, 104 |
| src/xngin/apiserver/worker/jobs/test\_autofail.py                                  |       68 |        2 |     97% |     69-70 |
| src/xngin/apiserver/worker/jobs/test\_snapshot\_analysis.py                        |      204 |        9 |     96% |57-63, 405-406 |
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
| **TOTAL**                                                                          | **20646** | **1103** | **95%** |           |

121 files skipped due to complete coverage.


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