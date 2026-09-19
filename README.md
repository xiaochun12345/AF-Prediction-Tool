# COPD–AF research calculator — revision_v2

This release updates the original Streamlit interface with the locked revised Elastic Net model. It uses 49 baseline predictors and averages probabilities from five separately fitted models (alpha 0.01, l1_ratio 0.1). It does not retrain a model when the website starts.

The study used UK Biobank internal and temporal validation. This research calculator estimates **net AF risk with death treated as censoring**, not competing-risk cumulative incidence. It has not been validated for clinical decision making.

## Run locally

Use Python 3.11 (Python 3.10 also tested in the study environment):

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Alternatively, run `python run_local.py --port 8501` and open `http://127.0.0.1:8501`. The launcher uses the certifi CA bundle if the local Windows certificate store is malformed; certificate and hostname verification remain enabled, and no system settings are changed.

Once dependencies are installed, prediction runs locally without accessing UK Biobank or any external model service. The application does not write submitted measurements to files or a database. On a hosted Streamlit deployment, inputs are processed by the host for the current session; do not enter personal identifiers.

## What changed

- The main model uses the 49 predictors selected in the revised analysis, including recorded COPD duration, 11 selected medication classes and the baseline inpatient J44.1 indicator. ICS was not selected. FVC, FEV1 and the supplied FEV1/FVC Z-score belong to a separate sensitivity analysis and are not main-model inputs.
- The study's exact scaling, one-hot encoding, constant-column handling, fitted coefficients, offsets and baseline hazards are compiled into `model.json` for all five models. No participant rows or imputation kernels are distributed.
- HbA1c uses mmol/mol; urate uses µmol/L; height uses metres. Falls have three categories. Insomnia coding is 0 = usually, 1 = never/rarely or sometimes. Smoking in the main model means current or previous smoking.
- `duration` means years from the UKB COPD report date (field 42016) to baseline, not verified biological disease duration. `j44_prior` means a first recorded inpatient J44.1 diagnosis on/before baseline, not exacerbation frequency.
- AF-PRS must use the same definition and scale as in the study. Other genetic scores are not interchangeable. All 49 inputs are required; the site does not replace missing information with normal values. The example uses marginal training medians/modes, not an actual participant.
- The main display reports 1-, 3-, 5- and 10-year probabilities. The old final-follow-up risk label and unvalidated high/medium/low categories are removed.
- CHARGE-AF uses the published five-year formula, with current smoking and antihypertensive treatment. HARMS₂-AF displays its published score only; the old locally refitted Cox curves for both scores are removed.

## Reproducibility

`verification.json` records comparisons against the five saved models and frozen predictions for all 2,091 internal and 1,417 temporal validation participants. Participants with complete raw inputs were also checked through the exact public inference function. For participants with missing measurements, parity was checked using each of the five corresponding completed datasets; the website itself requires complete inputs.

`inference.py` uses Python's standard library. `model.json` contains the source model SHA-256 and version. `example.json` is synthetic. `test_web.py` checks invalid inputs, score boundaries, application flows and stale-result handling:

```sh
python -m pip install pytest
python -m pytest test_web.py -q
```

`export_revision_model.py` and `feature_schema.py` describe the export performed within the author's analysis workspace; model regeneration requires the fitted revision_v2 models and scientific Python dependencies. They are not needed to run the calculator. The original analysis data are not included.

## Deployment

The application entry point is `app.py`; deploy the repository root with `requirements.txt` using Streamlit Community Cloud. Updating the existing repository/branch updates the existing application URL after deployment completes. Confirm the page displays **revision_v2_2026-09-19** before citing the updated application in the manuscript.

## References

- [CHARGE-AF original formula](https://pmc.ncbi.nlm.nih.gov/articles/PMC3647274/)
- [HARMS₂-AF publication](https://academic.oup.com/eurheartj/article/44/36/3443/7205602)
- [UKB COPD report date, field 42016](https://biobank.ndph.ox.ac.uk/ukb/field.cgi?id=42016)
- [UKB HbA1c, field 30750](https://biobank.ndph.ox.ac.uk/ukb/field.cgi?id=30750)
- [UKB urate, field 30880](https://biobank.ndph.ox.ac.uk/ukb/field.cgi?id=30880)
