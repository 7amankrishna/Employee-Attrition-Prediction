# Datasets

Four datasets are used by the analysis pipeline in `../analysis/` and the Python
backend in `../backend/`. Three of them are **real-world** datasets located on
GitHub (fetched on 2026-11-01); the fourth is the synthetic IBM sample that the
frontend already bundles.

| File | Rows | Target | Origin | Real? |
| --- | --- | --- | --- | --- |
| `../public/data/ibm-hr.csv` | 1,470 employees | `Attrition` (Yes/No) | IBM HR Analytics Employee Attrition & Performance (Kaggle), mirrored in many GitHub repos — **fictional data created by IBM data scientists** | No (synthetic benchmark) |
| `hr-comma-sep.csv` | 14,995 employees | `left` (1/0) | The classic anonymized **real company HRIS** export distributed as Kaggle "HR Analytics" (`HR_comma_sep.csv`); mirror: `github.com/KunalKurve/Machine-Learning` (`Cases/human-resources-analytics/HR_comma_sep.csv`) | Yes |
| `av-job-change-train.csv` | 18,359 candidates | `target` (1 = looking for a job change) | Analytics Vidhya JanataHack "HR Analytics: Job Change of Data Scientists" hackathon training data — real registrants of a data-science training program; mirror: `github.com/aasu14/Data-Science-Hackathon-And-Competition` (`AnalyticsVidhya JanataHack HR Analytics/train_jqd04QH.csv`) | Yes |
| `academic-faculty.csv` | 6,793 faculty | left academia between 2011 and 2017 (blank `title_2017`) | Supplementary data of Morgan, Way & Clauset (2018), *"Automatically assembling a full census of an academic field"*, arXiv:1804.02760 — `github.com/allisonmorgan/academic_census` (`data/faculty.csv`). A full census of one academic field (physics) at US PhD-granting departments | Yes (published research data) |
| `academic-department-attrition.csv` | 205 departments | `attrition_frac` | Same source, department-level attrition rates (`data/attrition.csv`) | Yes (published research data) |

## Schema notes

- `hr-comma-sep.csv`: `satisfaction_level` (0–1), `last_evaluation` (0–1),
  `number_project`, `average_montly_hours`, `time_spend_company` (years),
  `Work_accident`, `left`, `promotion_last_5years`, `Department`, `salary`
  (low/medium/high).
- `av-job-change-train.csv`: `enrollee_id`, `city` (anonymized), 
  `city_development_index` (0–1), `gender`, `relevent_experience`,
  `enrolled_university`, `education_level`, `major_discipline`, `experience`
  (years, may be `>20`/`<1`), `company_size`, `company_type`, `last_new_job`
  (years since last job change; `never`/`>4` allowed), `training_hours`,
  `target`.
- `academic-faculty.csv`: `title_2011`, `place_2011`, `title_2017`,
  `place_2017` (blank = no longer in the field), `sex` (M/F), `guid`
  (anonymized person id).

## Re-fetching

`fetch_datasets.py` re-downloads the three external files from their GitHub
mirrors (via `codeload.github.com`) and verifies the SHA-256 checksums recorded
below. Run it from the repository root:

```sh
python3 data/fetch_datasets.py          # downloads + verifies
```

## Checksums (SHA-256, 2026-11-01)

```
2f7028186426cdf4924d4c7b262c3f83bbca85eb14288add46b120d740e6a03b  data/hr-comma-sep.csv
591b300fcc58356f4b142f4b978ad42168f0a76663596d086a9592ccbb3a946c  data/av-job-change-train.csv
99fd9cecd66799566df9967544a85c45dee91779119ad3745bec0a38572e6ede  data/academic-faculty.csv
6a93edbc7ae9f9f737bfbe02734421e66eee549af85d5ca02de46e617a88a435  data/academic-department-attrition.csv
0d8e3f875d54a05d575f20344e6bf72c2fa3e2c86b825d342c89b4e122b1e300  public/data/ibm-hr.csv
```

## Licensing / use notes

- The IBM file is fictional and was created for analytics training; the Kaggle
  page states it is not real employee data.
- `HR_comma_sep.csv` is a widely redistributed anonymized HR export; treat as
  illustrative of a real company HRIS, not as identifiable personal data.
- The Analytics Vidhya hackathon data is anonymized candidate data shared for a
  public competition.
- The academic census is published research supplementary data (arXiv preprint);
  person-level rows are pseudonymized (`guid`).

None of these datasets contain names, contact details, or other direct
identifiers. They are used here for an educational retention-intelligence
demonstration, **not** for employment decisions about real people.
