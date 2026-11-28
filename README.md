# AI Facility Planning & Performance Intelligence

Decision support prototype for facility management: SLA breach prediction, facility and contract risk,
contractor analytics, a rule-based assistant and a human-approved decision memo. Arabic / English.

**All bundled data is synthetic.** It represents no real organisation, contractor, facility or contract.
Results demonstrate the method only.

## My data
A work order history (CSV) can be uploaded; every page then shows it for that browser session only. The app does
not save the file. This is a public prototype on an external server: do not upload confidential or real company data
without authorisation. Try it with `samples/history_sample.csv` (synthetic).

## Run
    pip install -r requirements.txt
    streamlit run app.py

## Limits
Alert precision and recall are modest, alerts are uneven across contractors, associations are not causes,
and no action should be taken without human review.
