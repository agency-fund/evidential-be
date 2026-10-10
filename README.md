[![test](https://github.com/agency-fund/evidential-be/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/agency-fund/evidential-be/actions/workflows/test.yaml)

# Evidential<a name="evidential"></a>

[Evidential](https://docs.evidential.dev/) is an experiments management platform built on FastAPI, Postgres, and React.

This repository contains the backend API server.

## Getting Started

See https://docs.evidential.dev/development/getting-started-dev/ for how to get started building and contributing.

## Google Sheets demos

Google Sheets is available as a **read-only datasource for preassigned A/B demos**. Users paste a public spreadsheet URL.
No Google API keys, OAuth, service account, or server credentials are needed. The connector downloads the linked tab's
public CSV export; assignment uploads are done manually by the presenter.

1. Import [the example CSV](tools/google_sheets_demo.csv) into a Google spreadsheet. This single sheet contains 1,000
   synthetic participant IDs, regions, and all seven metric options below. Six metrics have sample observations for
   planning and an immediate analysis; `onboarded_within_1_week` starts entirely blank so you can fill it live.
1. Set sharing to **Anyone with the link → Viewer** (Editor also works) and allow viewers to download. Copy the intended
   tab's normal URL, including `gid`. Without `gid`, the first tab is used. Public access must be allowed by your Workspace.
1. In Evidential, add a **Google Sheets (demo)** datasource. Paste the URL and select `linked_sheet`, which represents
   the single tab in that URL. Row 1 must contain
   unique column names using letters, numbers, and underscores. Use `participant_id` as the ID and choose a primary
   metric from the options below. Additional metrics can be selected as secondary outcomes.
   Leave **Cluster key** empty for individual assignment; use `region` under **Strata** to balance regions across arms.
   Choosing `region` as the cluster key instead assigns entire regions together (four clusters in this example).
1. Create and save a preassigned A/B experiment. Click **Download Experiment CSV** on the experiment page. It includes
   the participant ID, selected metrics, any fields used for filters, strata, or clustering, and
   `evidential_<experiment_id>_arm`, with arm names matched by participant ID. Values in these columns are retained;
   pending outcomes stay blank. Unassigned participants are retained with a blank arm. Keep the original setup tab.
   Use **File → Import → Upload → Insert new sheet(s)** to import the CSV, then rename the new tab **Experiment**.
   Disable **Convert text to numbers, dates, and formulas** to preserve participant IDs, including leading zeroes.
   [Google's import instructions](https://support.google.com/docs/answer/40608?hl=en) describe importing CSVs.
1. Open the **Experiment** tab, copy its URL including `gid`, and click **Connect Experiment tab** on the experiment
   page. Paste this URL into **Spreadsheet URL** and save. Each demo datasource reads one linked tab; updating its
   URL affects all experiments on that datasource. Use separate datasources for independent demos.
1. Fill or edit your selected metric columns in the **Experiment** tab, then click **Refresh** to update the live arm comparison.
   Each demo refresh saves an analysis snapshot and updates the live comparison and time-series chart. Saved history
   remains after reloading; demo chart points retain their exact timestamps rather than grouping by day.
   The button shows progress while saving and displays the last successful refresh time. For automatic updates,
   click **Start live demo**. Refreshes run about every 10 seconds, stop after 15 minutes, and stop when the page is closed or on an error.
   **Stop live demo** ends refreshes immediately. Google may briefly cache CSV exports, so edits can take longer to appear.
   Edit cells after connecting. Replacing or deleting a tab can change its `gid`; reconnect its current URL if this happens.

Each row represents one participant. Pick the metric that matches the organization you're demonstrating to:

| Sheet column                   | What it measures                                                                                                             | Example organization     |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| `minutes_on_site_last_7_days`  | Total minutes spent on site per participant over the last 7 days                                                             | Website or app           |
| `customer_satisfaction_1_to_5` | Satisfaction rating from 1 to 5                                                                                              | Customer experience team |
| `revenue_last_7_days`          | Revenue per participant over the last 7 days, using a consistent currency                                                    | Commerce or fundraising  |
| `purchases_last_7_days`        | Number of purchases per participant over the last 7 days                                                                     | Commerce                 |
| `support_resolution_hours`     | Hours to resolve a participant's support request; lower is better                                                            | Support team             |
| `assessment_score_0_to_100`    | Assessment score from 0 to 100                                                                                               | Education or training    |
| `onboarded_within_1_week`      | One-time onboarding outcome: blank while unknown, 1 if onboarded within 7 days, 0 once the window elapsed without onboarding | Onboarding or activation |

All metric options belong in the same sheet; pick a primary and optional secondary metrics for each experiment.
The onboarding column uses the connector's existing numeric handling of 1 and 0. Its mean is the onboarding rate
(for example, 0.75 represents 75% of participants with observed results). Keep pending results blank, including for
participants whose seven-day window is still open. Filling this column later does not require a new datasource.

Evidential reads the numbers you enter; calculate measures such as weekly minutes in the sheet or upstream. During
setup, the selected column's mean and spread provide the baseline statistics for sample-size planning. The default
**Minimum Effect** of 10% is the relative change you want to detect. During analysis, the baseline arm is the control
group. Filled demo values are synthetic examples, not measurements of a real intervention.

**Starting with blank outcomes:** keep participant IDs and metric headers populated, and leave the outcome cells empty.
Select your metric, click **Estimate Sample Size**, then choose **Use the maximum available sample size** (1,000 for the
starter) or a custom size. You can continue with unavailable power estimates for this POC; empty outcomes cannot provide
a meaningful power or minimum-detectable-effect estimate. Results appear as you add observed values to the assigned
participants; an arm comparison needs observations in both arms. Do not replace blanks with zero unless zero is the
actual observed outcome. The filled starter is easier if you want to demonstrate power analysis too.

For clustered assignment, a blank outcome still allows the connector to count participants and calculate cluster
sizes, but ICC and power estimates remain unavailable. Choose a maximum or custom cluster count to continue the demo.

This connector reads a fresh, temporary in-memory snapshot for each analysis using the existing warehouse query path.
The demo allows up to 5,000 data rows, 100 columns including assignment columns, and an 8 MB CSV download. Use plain
numeric outcome cells (no currency symbols or thousands separators); TRUE/FALSE are inferred as booleans. Blank cells are
missing outcomes, while zero and FALSE are observed outcomes. Entirely blank metric columns
are inferred as numeric. Format participant IDs
as plain text to preserve leading zeroes. Keep IDs unique and unchanged after assignment. Sorting rows between refreshes
is supported; newly added participants are not automatically enrolled in a preassigned experiment.

Live demo refreshes run while the experiment page is open and update the analysis on screen; historical snapshots keep
their existing schedule. Download the CSV immediately before importing it, since it contains a snapshot of the sheet.
