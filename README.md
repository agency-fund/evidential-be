# xngin

Python version of [RL Experiments Engine](https://github.com/agency-fund/rl-experiments-engine).

The following is a proposal of the main components of this service:

1. A ODBC/DBI-based interfae module to connect to underlying data sources (one table per unit of analysis)
2. A configuration module that draws from the table(s) specified in (1) into a Google Sheet that can be annotated with filters, metrics and strata
3. API endpoints that provide a list of fields and their values (/filters, /metrics, /strata)
4. API endpoints that provide a power analysis check and stratified random assignment of treatment
5. A treatment assignment service that stores treatment assignments and provides an API endpoint to provide treatment status by ID
6. Save experiment (inclusive of Audience) specifications 


