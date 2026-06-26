
```bash
head -2 Datasets/train.csv
```
Here is a description of the columns in `Datasets/train.csv` along with their likely data types, based on the header and a sample row:

1. **case_id**: Integer — Unique identifier for each case.
2. **Hospital_code**: Integer — Encoded hospital identifier.
3. **Hospital_type_code**: String — Encoded type of hospital (e.g., 'c').
4. **City_Code_Hospital**: Integer — Encoded city code for the hospital.
5. **Hospital_region_code**: String — Encoded region code (e.g., 'Z').
6. **Available Extra Rooms in Hospital**: Integer — Number of extra rooms available.
7. **Department**: String — Department name (e.g., 'radiotherapy').
8. **Ward_Type**: String — Encoded ward type (e.g., 'R').
9. **Ward_Facility_Code**: String — Encoded facility code (e.g., 'F').
10. **Bed Grade**: Float (nullable) — Bed grade, may have missing values.
11. **patientid**: Integer — Encoded patient identifier.
12. **City_Code_Patient**: Float (nullable) — Encoded city code for patient, may have missing values.
13. **Type of Admission**: String — Admission type (e.g., 'Emergency').
14. **Severity of Illness**: String — Severity level (e.g., 'Extreme').
15. **Visitors with Patient**: Integer — Number of visitors.
16. **Age**: String — Age range (e.g., '51-60').
17. **Admission_Deposit**: Float — Deposit amount for admission.
18. **Stay**: String — Target variable, length of stay category (e.g., '0-10').

