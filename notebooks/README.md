# Healthcare Length of Stay - Jupyter Notebooks

This directory contains comprehensive Jupyter notebooks for the Healthcare Length of Stay prediction project. These notebooks provide step-by-step walkthroughs of the entire machine learning pipeline, from data analysis to model deployment.

## 📚 Notebook Overview

### 1. [01_Healthcare_LOS_Complete_Pipeline.ipynb](01_Healthcare_LOS_Complete_Pipeline.ipynb)
**Complete ML Pipeline Walkthrough**
- **Purpose**: End-to-end machine learning pipeline demonstration
- **Content**: 
  - Data loading and validation
  - Preprocessing and feature engineering
  - Model training with multiple algorithms
  - Model evaluation and comparison
  - Results visualization and reporting
- **Best for**: Understanding the complete workflow
- **Audience**: Data scientists, ML engineers, stakeholders

### 2. [02_Data_Analysis_and_Preprocessing.ipynb](02_Data_Analysis_and_Preprocessing.ipynb)
**Exploratory Data Analysis & Preprocessing**
- **Purpose**: Deep dive into data understanding and preparation
- **Content**:
  - Comprehensive EDA with visualizations
  - Data quality assessment
  - Missing value analysis
  - Feature engineering and selection
  - Data preprocessing pipeline
- **Best for**: Data exploration and understanding
- **Audience**: Data analysts, data scientists

### 3. [03_Model_Training_and_Evaluation.ipynb](03_Model_Training_and_Evaluation.ipynb)
**Model Training & Performance Evaluation**
- **Purpose**: Focused model training and evaluation
- **Content**:
  - Multiple model training (XGBoost, LightGBM, CatBoost, Neural Networks)
  - Cross-validation strategies
  - Hyperparameter tuning
  - Performance metrics and comparison
  - Model selection
- **Best for**: Model development and comparison
- **Audience**: ML engineers, data scientists

### 4. [04_Model_Interpretation_and_Insights.ipynb](04_Model_Interpretation_and_Insights.ipynb)
**Model Interpretation & Business Insights**
- **Purpose**: Understanding model behavior and extracting insights
- **Content**:
  - Feature importance analysis
  - SHAP (SHapley Additive exPlanations) analysis
  - LIME explanations
  - Model behavior interpretation
  - Business insights and recommendations
- **Best for**: Model interpretability and business insights
- **Audience**: Business stakeholders, data scientists, domain experts

### 5. [05_Model_Prediction_Demo.ipynb](05_Model_Prediction_Demo.ipynb)
**Production Prediction Demonstration**
- **Purpose**: Demonstrating model usage for predictions
- **Content**:
  - Model loading and setup
  - Single patient predictions
  - Batch predictions
  - Prediction confidence analysis
  - Output formatting for different use cases
- **Best for**: Production deployment demonstration
- **Audience**: ML engineers, software developers

## 🚀 Getting Started

### Prerequisites
```bash
# Install required packages
pip install -r requirements.txt

# Ensure Jupyter is installed
pip install jupyter notebook

# For advanced visualizations
pip install plotly
```

### Running the Notebooks

1. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Navigate to the notebooks directory**

3. **Run notebooks in order** (recommended):
   - Start with `01_Healthcare_LOS_Complete_Pipeline.ipynb` for overview
   - Use `02_Data_Analysis_and_Preprocessing.ipynb` for detailed EDA
   - Continue with `03_Model_Training_and_Evaluation.ipynb` for model development
   - Use `04_Model_Interpretation_and_Insights.ipynb` for interpretability
   - Finish with `05_Model_Prediction_Demo.ipynb` for prediction examples

### Configuration

Each notebook will attempt to load configuration from `config/config.yaml`. If not found, default configurations are used for demonstration purposes.

## 📊 Key Features

### Interactive Visualizations
- Plotly-based interactive charts
- Seaborn statistical plots
- Custom visualization functions
- Real-time model performance tracking

### Comprehensive Analysis
- Statistical summaries
- Data quality reports
- Model performance metrics
- Feature importance analysis
- Business impact assessment

### Educational Content
- Step-by-step explanations
- Code comments and documentation
- Best practices demonstrations
- Error handling examples

## 🔧 Customization

### Adapting for Your Data
1. Update the data loading sections with your file paths
2. Modify feature engineering based on your dataset
3. Adjust model parameters in the configuration
4. Update target variable names and classes

### Adding New Models
1. Import new model classes in the training notebook
2. Add model configuration to the config dictionary
3. Include in the model training loop
4. Update evaluation and comparison sections

## 📈 Expected Outputs

### Visualizations
- Data distribution plots
- Correlation matrices
- Model performance comparisons
- Feature importance charts
- Confusion matrices
- ROC curves

### Reports
- Data quality summaries
- Model performance metrics
- Cross-validation results
- Feature importance rankings
- Business insights and recommendations

## 🤝 Contributing

When adding new notebooks or modifying existing ones:

1. Follow the established naming convention
2. Include comprehensive markdown explanations
3. Add error handling for missing data/models
4. Update this README with new notebook descriptions
5. Test with sample data if real data is not available

## 📝 Notes

- Notebooks are designed to be educational and production-ready
- Error handling is included for common issues (missing files, etc.)
- Sample data generation is provided when real data is unavailable
- All notebooks include progress indicators and status messages
- Code is optimized for both learning and practical use

## 🆘 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed and the `src` directory is in the path
2. **Data Loading Issues**: Check file paths and permissions
3. **Memory Issues**: Consider using data sampling for large datasets
4. **Model Loading**: Ensure models are trained and saved before running prediction notebooks

### Getting Help

- Check the individual notebook documentation
- Review the source code in the `src` directory
- Consult the main project README
- Check configuration files for parameter details 