# Repository Coverage



| Name                                                                 |    Stmts |     Miss |   Cover |   Missing |
|--------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/xngin/apiserver/apikeys.py                                       |       39 |        7 |     82% |37-41, 72-75 |
| src/xngin/apiserver/certs/certs.py                                   |       16 |        9 |     44% |21-25, 39-46 |
| src/xngin/apiserver/common\_field\_types.py                          |       12 |        1 |     92% |        13 |
| src/xngin/apiserver/conftest.py                                      |      193 |       14 |     93% |61, 80, 97-99, 109, 135, 137, 141, 289-290, 296-298, 316 |
| src/xngin/apiserver/customlogging.py                                 |       54 |        5 |     91% |41-42, 47, 62-63 |
| src/xngin/apiserver/database.py                                      |       61 |       16 |     74% |28, 42, 59, 65, 72, 93-109 |
| src/xngin/apiserver/dependencies.py                                  |       41 |       11 |     73% |26, 34, 61, 70-72, 79, 89, 99-101 |
| src/xngin/apiserver/dns/safe\_resolve.py                             |       44 |        9 |     80% |28-29, 33-34, 47, 74, 77, 82-83 |
| src/xngin/apiserver/dwh/dwh\_session.py                              |      179 |       73 |     59% |37, 43, 71-76, 79, 142, 150, 167-178, 185-246, 342-349, 352, 356-362, 380, 414-421, 429 |
| src/xngin/apiserver/dwh/inspection\_types.py                         |       61 |        5 |     92% |41, 61, 85, 96, 104 |
| src/xngin/apiserver/dwh/inspections.py                               |       28 |        1 |     96% |        65 |
| src/xngin/apiserver/dwh/queries.py                                   |      166 |       13 |     92% |130, 167-168, 186, 198, 247, 286, 338, 359, 378, 387, 389, 406 |
| src/xngin/apiserver/dwh/test\_dialect\_sql.py                        |       77 |        6 |     92% |535, 549, 552-557 |
| src/xngin/apiserver/dwh/test\_queries.py                             |      224 |        1 |     99% |       219 |
| src/xngin/apiserver/exceptionhandlers.py                             |       49 |       10 |     80% |31, 37, 47, 57-62, 76, 92 |
| src/xngin/apiserver/main.py                                          |       32 |        3 |     91% | 20-22, 51 |
| src/xngin/apiserver/models/storage\_format\_converters.py            |       91 |       16 |     82% |60, 77, 91-92, 113, 166-198, 292, 326-365 |
| src/xngin/apiserver/models/tables.py                                 |      272 |       16 |     94% |55, 168-170, 323, 345, 369, 374-376, 379-380, 498, 501, 506, 511 |
| src/xngin/apiserver/openapi.py                                       |       20 |        9 |     55% |     19-75 |
| src/xngin/apiserver/routers/admin/admin\_api.py                      |      492 |      247 |     50% |160, 233-238, 265-270, 295-300, 314-318, 331-337, 360, 382, 393, 408-421, 441-451, 483-500, 518-535, 570-582, 586-599, 621-641, 658-677, 692-698, 715-725, 757-768, 796-822, 833-851, 861-865, 884-891, 897-905, 927-941, 955, 1009-1019, 1034-1121, 1133, 1148-1173, 1200-1206, 1260-1265, 1320-1335, 1360-1393, 1452, 1479-1480, 1494-1495, 1506, 1518-1526, 1537-1543, 1560-1567, 1599-1610, 1651-1665 |
| src/xngin/apiserver/routers/admin/admin\_api\_types.py               |      144 |        2 |     99% |    31, 33 |
| src/xngin/apiserver/routers/admin/generic\_handlers.py               |       24 |       14 |     42% |     41-59 |
| src/xngin/apiserver/routers/assignment\_adapters.py                  |       45 |        1 |     98% |       161 |
| src/xngin/apiserver/routers/auth/auth\_api.py                        |       32 |       10 |     69% |26, 30, 61-84 |
| src/xngin/apiserver/routers/auth/auth\_dependencies.py               |       96 |       47 |     51% |56, 71-79, 86-117, 137, 144-187, 218-220 |
| src/xngin/apiserver/routers/common\_api\_types.py                    |      329 |       44 |     87% |106, 240, 310-319, 339, 550-554, 595, 766-789, 796-806, 813-819, 1016 |
| src/xngin/apiserver/routers/common\_enums.py                         |      169 |       45 |     73% |73, 75, 87, 89, 98-107, 112, 137, 150-151, 172, 209, 247-260, 280-286, 314-318 |
| src/xngin/apiserver/routers/experiments/dependencies.py              |       18 |        3 |     83% |     51-57 |
| src/xngin/apiserver/routers/experiments/experiments\_api.py          |       64 |       15 |     77% |101, 110-123, 147, 160, 187, 198, 212, 236-241, 258-265 |
| src/xngin/apiserver/routers/experiments/experiments\_common.py       |      293 |       86 |     71% |69, 74, 115, 123, 148-158, 180, 205, 221-230, 254, 265, 335, 349, 374-392, 424-442, 489-496, 552-570, 624-626, 703, 707, 730, 753, 766, 794-882 |
| src/xngin/apiserver/routers/experiments/test\_experiments\_common.py |      338 |        1 |     99% |       933 |
| src/xngin/apiserver/routers/healthchecks\_api.py                     |       16 |        2 |     88% |     26-29 |
| src/xngin/apiserver/routers/proxy\_mgmt/proxy\_mgmt\_api.py          |       50 |        7 |     86% |52, 119-126 |
| src/xngin/apiserver/routers/proxy\_mgmt/test\_proxy\_mgmt\_api.py    |       63 |        1 |     98% |        38 |
| src/xngin/apiserver/routers/stateless/stateless\_api.py              |       89 |        3 |     97% |215, 256, 316 |
| src/xngin/apiserver/routers/stateless/stateless\_api\_types.py       |       43 |        7 |     84% |     38-45 |
| src/xngin/apiserver/routers/stateless/test\_stateless\_api.py        |       75 |        8 |     89% |120, 124-125, 160-164 |
| src/xngin/apiserver/routers/test\_common\_api\_types.py              |       41 |        1 |     98% |        76 |
| src/xngin/apiserver/settings.py                                      |      220 |       36 |     84% |61-65, 150, 199, 206, 212, 272, 274-280, 342, 349, 352-357, 411-417, 438, 460, 471, 475, 485, 488, 516, 560 |
| src/xngin/apiserver/settings\_secrets.py                             |       61 |       44 |     28% |24, 30-34, 40-45, 51-59, 65-84, 90-99, 104-112 |
| src/xngin/apiserver/testing/assertions.py                            |       14 |        2 |     86% |    17, 27 |
| src/xngin/cli/main.py                                                |      388 |      240 |     38% |96-112, 130-138, 149-181, 185-189, 201-203, 298, 301, 306, 318-319, 328, 336-354, 357-393, 395-398, 418-423, 428-429, 485-542, 565-576, 582-588, 594-597, 606-614, 639-644, 659-667, 680-688, 692-695, 737-824, 842-846, 859-860, 870-872, 876 |
| src/xngin/db\_extensions/custom\_functions.py                        |       41 |        3 |     93% |47, 71, 84 |
| src/xngin/db\_extensions/test\_custom\_functions.py                  |       48 |       12 |     75% |59-68, 73-82 |
| src/xngin/events/common.py                                           |        7 |        2 |     71% |    14, 21 |
| src/xngin/events/experiment\_created.py                              |       14 |        4 |     71% | 18, 21-25 |
| src/xngin/events/webhook\_sent.py                                    |       15 |        5 |     67% |     18-23 |
| src/xngin/sheets/config\_sheet.py                                    |       73 |       22 |     70% |23-33, 40-41, 44, 79, 83, 86, 102-103, 110-112, 115 |
| src/xngin/sheets/gsheets.py                                          |       23 |       13 |     43% |15, 24-37, 42, 50-51 |
| src/xngin/sheets/test\_gsheets.py                                    |       22 |        9 |     59% |     13-26 |
| src/xngin/stats/balance.py                                           |       62 |        3 |     95% |113, 143, 146 |
| src/xngin/stats/bandit\_sampling.py                                  |       70 |       59 |     16% |27-29, 50-57, 76-78, 100-115, 141-164, 183-215, 237-277 |
| src/xngin/stats/power.py                                             |       65 |        1 |     98% |        94 |
| src/xngin/xsecrets/gcp\_kms\_provider.py                             |       71 |       26 |     63% |62-79, 86-89, 106-107, 110, 114-122, 126-133 |
| src/xngin/xsecrets/provider.py                                       |       19 |        1 |     95% |        46 |
| src/xngin/xsecrets/secretservice.py                                  |       61 |        4 |     93% |42-43, 106, 130 |
| src/xngin/xsecrets/test\_gcp\_kms\_provider.py                       |      103 |       26 |     75% |40-42, 170-175, 182-189, 195-203, 210, 217-228 |
| src/xngin/xsecrets/test\_nacl\_provider.py                           |       67 |        1 |     99% |        24 |
|                                                            **TOTAL** | **7464** | **1282** | **83%** |           |

62 files skipped due to complete coverage.


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://github.com/agency-fund/evidential-be/raw/python-coverage-comment-action-data/badge.svg)](https://github.com/agency-fund/evidential-be/tree/python-coverage-comment-action-data)

This is the one to use if your repository is private or if you don't want to customize anything.



## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.