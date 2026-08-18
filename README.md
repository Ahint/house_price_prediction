# 🏡 Real Estate Valuation AI

An intelligent, machine learning-powered web application for predicting real estate property prices based on 13 key characteristics. Built with **Streamlit** and powered by a trained ML model.

---

## ✨ Features

### 🎯 **Valuation Tab**
- **Instant Price Prediction** - Get real-time property valuations as you adjust parameters
- **Quick Settings** - Fast access to 4 most important features (Rooms, Crime, Age, Income %)
- **Advanced Settings** - Fine-tune all 13 property characteristics
- **Visual Profile** - Radar chart showing your property across all dimensions
- **Market Comparison** - See how your property compares to average market value
- **Input Summary** - Toggle-able detailed input review with normalization percentages

### 📊 **Analytics Tab**
- **Feature Sensitivity Analysis** - Interactive line chart showing price impact of any feature
- **Scenario Comparison** - Compare up to 3 different property scenarios side-by-side
- **Feature Impact Ranking** - Horizontal bar chart showing which features matter most
- **Market Distribution** - Histogram of 500 similar properties showing market positioning
- **Statistical Summary** - Mean, median, min, max price analysis

### ℹ️ **Information Tab**
- **About the Tool** - How the application works
- **Disclaimer** - Important limitations and considerations
- **Feature Descriptions** - Detailed explanation of all 13 input features

---

## 📋 Input Features

The app analyzes 13 key property characteristics:

| Feature | Unit | Range | Category |
|---------|------|-------|----------|
| **CRIM** - Crime Rate | per capita | 0-30 | Location |
| **ZN** - Residential Zoning | % of land | 0-100 | Location |
| **INDUS** - Non-Retail Business | % acres | 0-30 | Building |
| **CHAS** - Charles River Proximity | binary | 0-1 | Location |
| **NOX** - Air Pollution | ppm/10 | 0.3-0.9 | Environment |
| **RM** - Rooms per Dwelling | count | 3-10 | Building |
| **AGE** - Building Age | % pre-1940 | 0-100 | Building |
| **DIS** - Distance to Jobs | weighted miles | 0-15 | Location |
| **RAD** - Highway Access | index (1-24) | 1-24 | Location |
| **TAX** - Property Tax | per $10k | 150-710 | Economic |
| **PTRATIO** - Student-Teacher Ratio | ratio | 12-22 | Economic |
| **B** - Demographics Index | index | 0-400 | Environment |
| **LSTAT** - Low Income Population | % residents | 0-40 | Economic |

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download
```bash
# If using git
git clone <repository-url>
cd real-estate-valuation

# Or download the files directly
```

### Step 2: Install Dependencies
```bash
pip install streamlit matplotlib seaborn scikit-learn joblib pandas numpy
```

Or install from requirements file (if available):
```bash
pip install -r requirements.txt
```

### Step 3: Prepare Model Files
Ensure you have the following files in the project directory:
- `model.joblib` - Trained ML model
- `scaler.joblib` - Feature scaler for data preprocessing

### Step 4: Run the Application
```bash
streamlit run app_v3.py
```

The app will open in your default browser at `http://localhost:8501`

---

## 📱 Usage Guide

### Getting Started

1. **Open the App** - Run the command above to start the application
2. **Adjust Quick Settings** - Use the 4 main sliders to set key parameters
3. **View Prediction** - See real-time price estimate update automatically

### Quick Settings (Left Panel)
- **Rooms** - Number of rooms in the property
- **Crime Rate** - Local crime statistics
- **Building Age** - Age of the structure
- **Low Income %** - Percentage of lower-income residents in area

### Advanced Settings (Expandable)
Click "⚙️ Advanced Settings" to access all 13 features:
- Zoning information
- Business density
- Pollution levels
- Job accessibility
- School quality
- Demographics
- Taxes
- And more...

### Understanding the Results

**Estimated Price** - The model's prediction based on current inputs
- **vs. Average** - Comparison to average property value
- **Key Metrics** - Detailed breakdown of valuation

### Analytics Tab

#### Feature Sensitivity
- Select any feature from dropdown
- See how price changes across the feature's full range
- Useful for understanding feature importance

#### Scenario Comparison
- Create 3 different property scenarios
- Adjust rooms and crime rate for each
- Compare prices side-by-side
- Ideal for "what-if" analysis

#### Feature Impact
- Ranks all 13 features by importance
- Green bars = positive impact, Red = negative impact
- Shows % change when feature is at maximum value
- Helps identify optimization opportunities

#### Market Distribution
- Shows price range of 500 similar properties
- Your property marked with golden line
- Statistical overview (mean, median, min, max)
- Identifies if your property is above/below average

---

## 🎨 Design & UI

### **Version Comparison**

| Feature | app.py | app_v2.py | app_v3.py |
|---------|--------|-----------|-----------|
| Design | Tabbed Layout | Two-Column | Two-Column |
| Theme | Modern Gradient | Card-Based | Card-Based |
| Color | Dark Blue/Teal | Blue/Purple | Blue/Purple |
| Graphs | N/A | Plotly | Matplotlib |
| Best For | Simple, Clean | Visual Analysis | Lightweight |

### Color Scheme
- **Primary**: Deep Blue (#1E40AF)
- **Secondary**: Purple (#7C3AED)
- **Accent**: Amber (#F59E0B)
- **Success**: Green (#10B981)
- **Danger**: Red (#EF4444)

---

## 📊 How It Works

### Machine Learning Pipeline

1. **Input Collection** - 13 property features collected via UI
2. **Feature Scaling** - Features normalized using StandardScaler
3. **Model Prediction** - Trained regression model generates price estimate
4. **Output Display** - Prediction shown with market context

### Model Details
- **Algorithm**: Regression (likely Linear or Random Forest based on training)
- **Features**: 13 standardized inputs
- **Output**: Predicted property price in USD
- **Accuracy**: Depends on training data quality

---

## 📁 File Structure

```
├── app.py              # Original light-mode version
├── app_v2.py           # Advanced version with Plotly graphs
├── app_v3.py           # Current version with Matplotlib (Recommended)
├── model.joblib        # Trained ML model (required)
├── scaler.joblib       # Feature scaler (required)
└── README.md           # This file
```

### File Descriptions

| File | Purpose |
|------|---------|
| `app.py` | Light-mode Streamlit app with clean tabbed interface |
| `app_v2.py` | Advanced analytics with Plotly visualizations |
| `app_v3.py` | Lightweight version with Matplotlib (No external deps) |
| `model.joblib` | Pickled ML model for price prediction |
| `scaler.joblib` | Pickled StandardScaler for feature normalization |

---

## 🔧 Customization

### Changing Features Range
Edit the `features_data` dictionary in the app file:

```python
features_data = {
    "RM": {
        "name": "Rooms per Dwelling",
        "unit": "count",
        "min": 3,  # Edit minimum
        "max": 10,  # Edit maximum
        "step": 0.1,  # Edit slider step
        "category": "Building",
    },
    # ... more features
}
```

### Changing Colors
Modify the CSS variables in the `<style>` section:

```css
:root {
    --primary: #1E40AF;    /* Change primary color */
    --secondary: #7C3AED;  /* Change secondary color */
    --accent: #F59E0B;     /* Change accent color */
}
```

### Default Input Values
Edit the session state initialization:

```python
if "inputs" not in st.session_state:
    st.session_state.inputs = {
        "CRIM": 5.0,  # Default crime rate
        "RM": 6.5,  # Default rooms
        # ... more defaults
    }
```

---

## 📈 Understanding the Graphics

### Property Profile Bar Chart
- Shows each feature's value as percentage of its range
- Purple = High value, Orange = Low value
- 100% = Feature at maximum, 0% = Feature at minimum

### Feature Sensitivity Curve
- X-axis: Feature value range
- Y-axis: Predicted price
- Shows non-linear relationships between features and price

### Scenario Comparison
- Compares 3 different property configurations
- Bar heights represent predicted prices
- Helps identify which changes have biggest impact

### Feature Impact Ranking
- Shows which features affect price most
- Positive (green) = Increases price
- Negative (red) = Decreases price
- Useful for property optimization strategies

### Price Distribution
- Market positioning analysis
- Your property vs. 500 similar properties
- Identifies pricing outliers

---

## ⚠️ Disclaimer & Limitations

**This tool is for estimation purposes only.**

### What This Tool DOES:
- ✅ Provide quick preliminary estimates
- ✅ Compare relative property values
- ✅ Analyze feature impacts
- ✅ Support scenario planning

### What This Tool DOES NOT:
- ❌ Replace professional appraisals
- ❌ Account for market fluctuations
- ❌ Factor in property condition/renovations
- ❌ Consider latest comparable sales
- ❌ Include time-sensitive market trends
- ❌ Account for location-specific amenities

### Important Notes:
- Predictions based on historical training data
- May not reflect current market conditions
- Always verify with professional appraisers
- Use as reference point, not sole decision factor
- Market conditions change rapidly

---

## 🔐 Data & Privacy

- **Local Processing**: All calculations done locally on your machine
- **No Data Transmission**: Input values not sent to external servers
- **Model Transparency**: Using standard ML algorithms

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'streamlit'"
**Solution**: Install Streamlit
```bash
pip install streamlit
```

### Issue: "FileNotFoundError: model.joblib not found"
**Solution**: Ensure model.joblib and scaler.joblib are in the same directory as the app

### Issue: "ValueError: could not convert string to float"
**Solution**: Check that all input values are numeric

### Issue: App runs but shows blank charts
**Solution**: Ensure matplotlib is installed
```bash
pip install matplotlib seaborn
```

### Issue: Slow performance
**Solution**: 
- Reduce number of scenarios analyzed
- Close other applications
- Use app_v3.py (lighter than v2)

---

## 📚 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | 1.0+ | Web app framework |
| pandas | 1.0+ | Data manipulation |
| numpy | 1.0+ | Numerical computing |
| matplotlib | 3.0+ | Chart visualization |
| seaborn | 0.11+ | Statistical graphics |
| scikit-learn | 0.24+ | ML algorithms |
| joblib | 1.0+ | Model persistence |

---

## 🚀 Performance Tips

1. **Faster Predictions** - Use app_v3.py instead of v2.py
2. **Reduce Visualizations** - Close unused tabs
3. **Fewer Scenarios** - Limit scenario analysis to 1-2 comparisons
4. **Local Storage** - Keep model.joblib and scaler.joblib in app directory

---

## 📝 Example Workflows

### Workflow 1: Property Valuation
1. Enter property characteristics in Quick Settings
2. Fine-tune with Advanced Settings
3. Check "View Summary" to see profile
4. Note the predicted price
5. Compare to market average in Key Metrics

### Workflow 2: Feature Impact Analysis
1. Go to Analytics tab
2. Use "Feature Sensitivity" to test each feature
3. Check "Feature Impact" ranking
4. Identify which improvements have best ROI

### Workflow 3: Scenario Planning
1. Navigate to Analytics → Scenario Comparison
2. Set up 3 property variations
3. Compare prices
4. Determine which changes most valuable
5. Optimize your property strategy

### Workflow 4: Market Research
1. Keep current property settings
2. Analyze "Price Distribution"
3. Check your position vs. market
4. Review statistical summary
5. Understand market pricing trends

---

## 📞 Support & Feedback

### Reporting Issues
- Check Troubleshooting section above
- Verify all dependencies installed
- Ensure model files present
- Check Python version (3.8+)

### Feature Requests
- Open an issue on GitHub
- Describe desired feature
- Include use case

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Visualizations with [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/)
- ML framework: [Scikit-learn](https://scikit-learn.org/)
- Model trained on real estate dataset

---

## 🎯 Quick Start Summary

```bash
# 1. Install dependencies
pip install streamlit matplotlib seaborn scikit-learn joblib pandas numpy

# 2. Ensure model files are in directory
# - model.joblib
# - scaler.joblib

# 3. Run the app
streamlit run app_v3.py

# 4. Open browser (auto opens) or go to
# http://localhost:8501
```

---

## 📊 Version History

### v3.0 (Current)
- ✅ Matplotlib graphs (no Plotly needed)
- ✅ Lighter weight
- ✅ Better performance
- ✅ All analytics features

### v2.0
- Plotly interactive charts
- Advanced analytics
- Scenario comparison

### v1.0
- Original light-mode design
- Basic input/output
- Clean interface

---

**Happy predicting! 🎉**

For questions or issues, please refer to the Troubleshooting section or contact support.

Last Updated: August 2026