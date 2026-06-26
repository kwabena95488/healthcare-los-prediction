# Healthcare Length of Stay EDA Summary

## Dataset Overview
- **Training set**: 318,438 records with 18 features
- **Test set**: 137,057 records with 17 features (excluding the target variable 'Stay')

## Target Variable Analysis
- The target variable 'Stay' represents the length of hospital stay categorized into 11 groups
- Most common stay durations:
  - 21-30 days: 27.48%
  - 11-20 days: 24.54%
  - 31-40 days: 17.32%
- Least common stay durations:
  - 61-70 days: 0.86%
  - 91-100 days: 0.87%
  - More than 100 days: 2.10%
- The distribution is imbalanced, which will need to be addressed during modeling

## Numerical Features Analysis
- **Hospital code**: 32 unique hospitals with codes 1-32
- **City Code Hospital**: 13 unique city codes
- **Available Extra Rooms**: Most hospitals have 2-4 extra rooms available
- **Bed Grade**: Ranges from 1-4, with grade 3 being the most common
- **Visitors with Patient**: Most patients have 2-4 visitors
- **Admission Deposit**: Ranges from 1,800 to 11,008 with most deposits around 4,200-5,400

## Categorical Features Analysis
- **Hospital Type**: 7 types (a-g) with type 'a' being most common (45.04%)
- **Hospital Region**: 3 regions (X, Y, Z) with region X (41.87%) and Y (38.45%) being dominant
- **Department**: Heavily skewed towards gynecology (78.35%)
- **Ward Type**: Dominated by types R (40.18%), Q (33.34%), and S (24.43%)
- **Ward Facility Code**: 6 codes (A-F) with code F being most common (35.41%)
- **Type of Admission**: Mainly Trauma (47.81%) and Emergency (36.95%)
- **Severity of Illness**: Mostly Moderate (55.22%) and Minor (26.97%)

## Age Feature Analysis
- Age is provided as categorical ranges
- Most patients are in the 31-60 age range (55.23%)
- When converted to numeric midpoints, the distribution appears bimodal with peaks around 35-45 and 65-75

## Missing Data Analysis
- Minimal missing data:
  - City_Code_Patient: 1.42% missing in training data, 1.57% in test data
  - Bed Grade: 0.035% missing in training data, 0.026% in test data
- These can be addressed using imputation techniques

## Key Relationships with Target Variable
1. **Age vs Stay**: Older patients tend to have longer stays
2. **Department vs Stay**: Gynecology and anesthesia have different stay patterns
3. **Type of Admission vs Stay**: Emergency admissions often have shorter stays than trauma
4. **Severity of Illness vs Stay**: Extreme cases show longer stays
5. **Visitors with Patient vs Stay**: More visitors correlate with shorter stays
6. **Hospital factors**: Hospital code, type, and region show varied patterns in stay duration

## Recommendations for Preprocessing and Modeling
1. **Handle missing values**: Use imputation for City_Code_Patient and Bed Grade
2. **Feature encoding**:
   - Convert Age ranges to numeric representation
   - Apply one-hot encoding for nominal categories
   - Consider target encoding for high-cardinality features
3. **Class imbalance**: Address through resampling or class weights
4. **Feature engineering opportunities**:
   - Create interaction terms between Department and Severity
   - Generate aggregated features based on Hospital and City codes
5. **Potential models**: Tree-based models likely to perform well given categorical nature of data

All visualizations have been saved in the `figures/` directory for further reference. 