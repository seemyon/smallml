# SmallML Business Quickstart

Predict which customers will churn using your existing customer data - even if you only have 50-200 customers.

**No machine learning expertise required.**

## What You Need

- Customer data in CSV format (or pandas DataFrame)
- A column indicating which customers churned (0 = stayed, 1 = left)
- At least 50 customers with known outcomes
- Python 3.9+

## Installation

```bash
pip install smallml
```

## Quick Start (5 Minutes)

### Option 1: Single Location (One Store/Business)

If you have one business location with customer data:

```python
from smallml import Pipeline
import pandas as pd

# Load your customer data
customers = pd.read_csv('my_customers.csv')

# Create and fit the pipeline
pipeline = Pipeline(use_pretrained_priors=True)
pipeline.fit(customers, target_col='churned')

# Predict on new customers
new_customers = pd.read_csv('new_leads.csv')
predictions = pipeline.predict(new_customers)

print(predictions)
```

### Option 2: Multiple Locations (Chain/Franchise)

If you have multiple stores or branches:

```python
from smallml import Pipeline
import pandas as pd

# Load data for each location
data = {
    'store_downtown': pd.read_csv('downtown_customers.csv'),
    'store_mall': pd.read_csv('mall_customers.csv'),
    'store_suburb': pd.read_csv('suburb_customers.csv'),
}

# Create and fit (combines data from all locations for better accuracy)
pipeline = Pipeline(use_pretrained_priors=True)
pipeline.fit(data, target_col='churned')

# Predict for a specific store
predictions = pipeline.predict(new_customers, sme_id='store_downtown')
```

## Understanding the Output

```
   prediction  bayesian_std  conformal_set  conformal_set_size
0        0.23          0.12            {0}                   1
1        0.78          0.15            {1}                   1
2        0.51          0.21          {0,1}                   2
```

| Column | Meaning |
|--------|---------|
| `prediction` | Probability of churn (0.0 to 1.0) |
| `bayesian_std` | Uncertainty in the prediction |
| `conformal_set` | **Decision guide** (see below) |
| `conformal_set_size` | 1 = confident, 2 = uncertain |

### Reading Conformal Sets

| Set | Meaning | Action |
|-----|---------|--------|
| `{0}` | Confident: Will NOT churn | Low priority |
| `{1}` | Confident: WILL churn | **Contact immediately!** |
| `{0,1}` | Uncertain: Could go either way | Monitor closely |

## Preparing Your Data

### Required Format

Your CSV should have:
- One row per customer
- Columns for customer attributes (features)
- One column indicating churn (0 or 1)

Example:
```csv
customer_id,days_since_purchase,purchase_count,total_spent,tenure_months,churned
1001,15,8,450.00,24,0
1002,45,2,89.50,6,1
1003,7,12,890.00,36,0
```

### Recommended Features

These features work well for churn prediction:

| Feature | Description |
|---------|-------------|
| `recency` | Days since last purchase/interaction |
| `frequency` | Number of purchases/visits |
| `monetary` | Total amount spent |
| `tenure` | How long they've been a customer |
| `age` | Customer age (if available) |

Don't worry about exact names - SmallML will match common variations automatically.

## Evaluating Your Model

```python
# Split your data: some for training, some for testing
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(customers, test_size=0.2)

# Fit on training data
pipeline = Pipeline(use_pretrained_priors=True)
pipeline.fit(train_data, target_col='churned')

# Evaluate on test data
X_test = test_data.drop('churned', axis=1)
y_test = test_data['churned']

metrics = pipeline.evaluate(X_test, y_test)

print(f"Accuracy: {metrics['accuracy']:.1%}")
print(f"AUC Score: {metrics['auc']:.3f}")
print(f"Coverage: {metrics['conformal_coverage']:.1%}")  # Should be ~90%
```

## Saving and Loading Your Model

```python
# Save for later use
pipeline.save('my_churn_model.pkl')

# Load when needed
from smallml import Pipeline
pipeline = Pipeline.load('my_churn_model.pkl')

# Use immediately
predictions = pipeline.predict(new_customers)
```

## Common Questions

**Q: How many customers do I need?**
- Minimum: 50 customers with known churn outcomes
- Recommended: 100+ for better accuracy
- Multiple locations help (data is combined intelligently)

**Q: How long does training take?**
- Single location: 15-30 minutes
- Quick mode (less accurate): 5-10 minutes

**Q: What if I don't have churn data yet?**
- You need historical data showing which customers left
- Start tracking now: flag customers who haven't returned in 90 days
- Come back when you have 50+ labeled examples

**Q: Can I use this for things other than churn?**
- Yes! Any binary prediction works (conversion, default, etc.)
- Just change `target_col` to your outcome column

**Q: My predictions are all uncertain ({0,1}). Why?**
- Your features may not be predictive enough
- Try adding more relevant features
- Or you may need more training data

## Getting Help

- **Examples**: See `examples/quickstart.py` and `examples/single_entity_quickstart.py`
- **Issues**: https://github.com/seemyon/smallml/issues
- **Full Documentation**: See the main README

## Next Steps

1. Start with your existing customer data
2. Train a model using the code above
3. Identify high-risk customers (conformal_set = {1})
4. Take action: reach out, offer incentives, understand why they're leaving
5. Retrain monthly as you collect more data
