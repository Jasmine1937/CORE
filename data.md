# Data sources and two-year targets

Build a company-year panel from annual financial records and dated distress events. Save the resulting table as `panel.csv` using the column schema below. The modeling workflow accepts the same schema for Chinese and U.S. data.

## China

Sources: [CSMAR](https://www.csmar.com/en/channels/77.html) and [CNRDS](https://www.cnrds.com/Home/About).

Download annual financial variables, stock codes, annual security names, and ST/*ST status records. Format stock codes as six-character strings. Join financial tables by stock code and financial year.

For a company in normal status in year `t`, pair its financial predictors with the company's status in year `t+2`. Assign target 1 to a qualifying new ST/*ST event in `t+2`, and target 0 to an observed normal status in that year. Consecutive ST years belong to one episode. A recurrent episode qualifies after at least three consecutive complete normal years preceding the new ST/*ST event. Track PT as a separate status category.

Construct the labeled panel from normal predictor years with observed target-year status. Omit PT, continuing distress episodes, and company-years with an unobserved target year.

## United States

Sources:

- [SEC Financial Statement and Notes Data Sets](https://www.sec.gov/data-research/sec-markets-data/financial-statement-notes-data-sets): reported financial statement data.
- [LoPucki Bankruptcy Research Database](https://lopucki.law.ufl.edu/): company bankruptcy events and filing dates.
- [SEC EDGAR](https://www.sec.gov/search-filings): original filings, Form 8-K disclosures, and exhibits. The [SEC APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) provide filing histories and structured company data.

Extract annual financial variables, financial years, and fiscal-year-end dates. Harmonize reporting scope and measurement units, and retain one consistent annual record per company and financial year. Format CIKs as ten-character strings.

Combine LoPucki and EDGAR bankruptcy records into an event table containing the pre-bankruptcy company's CIK and the actual bankruptcy petition date. Resolve company identity and duplicate events using a consistent company-level event definition, then join events to financial records by CIK.

For fiscal-year-end date `d` and bankruptcy petition date `e`, assign target 1 when an event satisfies:

```text
0 < (e - d).days <= 730
```

Assign target 0 when the complete 730-day interval is observed and contains zero qualifying events. Company-years with incomplete event follow-up remain outside the labeled panel.

## Shared input schema

| Column | Content |
|---|---|
| `entity_id` | Company identifier as a string |
| `year` | Financial year as an integer |
| `target` | Observed two-year binary target |
| `zscore` | Precomputed Z-score; higher values indicate better financial health |
| Financial predictors | Numeric variables available at the predictor date |

Map existing column names through `config.json`. Specify the candidate financial columns with `data.features`; the Z-score column serves the Z-score baseline.

Store predictors in their original measurement units, using plain numeric values and empty cells for missing observations. Use a consistent ratio convention within each column, such as `0.12` for 12%. Preprocessing parameters are learned within each source training window, followed by feature-specific transformation and training-set SMOTEENN. Validation and final-test cohorts retain their observed rows and class distributions.
