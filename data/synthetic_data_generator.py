import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_customer_data(n_customers=5000):
    """
    Generate synthetic customer data for churn prediction analysis
    """
    
    # Customer Demographics
    customer_ids = [f"CUST_{str(i).zfill(6)}" for i in range(1, n_customers + 1)]
    
    # Age distribution (realistic for utility company)
    ages = np.random.normal(45, 15, n_customers)
    ages = np.clip(ages, 18, 85).astype(int)
    
    # Gender distribution
    genders = np.random.choice(['Male', 'Female', 'Other'], n_customers, p=[0.48, 0.48, 0.04])
    
    # Location (UK regions)
    regions = np.random.choice([
        'London', 'Birmingham', 'Manchester', 'Glasgow', 'Liverpool', 
        'Bristol', 'Sheffield', 'Leeds', 'Edinburgh', 'Cardiff'
    ], n_customers, p=[0.15, 0.12, 0.10, 0.08, 0.08, 0.07, 0.07, 0.07, 0.06, 0.20])
    
    # Tenure (months with company)
    tenure_months = np.random.exponential(24, n_customers)
    tenure_months = np.clip(tenure_months, 1, 120).astype(int)
    
    # Monthly charges (£)
    base_charge = np.random.normal(85, 25, n_customers)
    monthly_charges = np.clip(base_charge, 25, 200).round(2)
    
    # Total charges (tenure * average monthly charge with some variation)
    total_charges = []
    for i in range(n_customers):
        avg_monthly = monthly_charges[i] * (1 + np.random.normal(0, 0.1))
        total = avg_monthly * tenure_months[i]
        total_charges.append(max(total, 0))
    
    # Service usage patterns
    avg_monthly_usage = np.random.gamma(2, 150, n_customers).astype(int)  # kWh
    
    # Contract type
    contract_types = np.random.choice(['Month-to-month', '1-year', '2-year'], 
                                    n_customers, p=[0.4, 0.35, 0.25])
    
    # Payment method
    payment_methods = np.random.choice([
        'Direct Debit', 'Credit Card', 'Bank Transfer', 'Cash/Cheque'
    ], n_customers, p=[0.45, 0.30, 0.20, 0.05])
    
    # Customer service interactions
    support_calls = np.random.poisson(2, n_customers)
    
    # Digital engagement score (0-100)
    digital_engagement = np.random.beta(2, 2, n_customers) * 100
    digital_engagement = digital_engagement.astype(int)
    
    # Satisfaction score (1-10)
    # Inversely related to support calls and positively to tenure
    base_satisfaction = 7 + (tenure_months / 120) * 2 - (support_calls * 0.3)
    satisfaction_scores = np.clip(np.random.normal(base_satisfaction, 1.5), 1, 10).round(1)
    
    # Generate churn based on realistic business logic
    churn_probability = []
    churned = []
    
    for i in range(n_customers):
        # Churn factors
        prob = 0.1  # Base probability
        
        # Tenure effect (newer customers more likely to churn)
        if tenure_months[i] < 6:
            prob += 0.4
        elif tenure_months[i] < 12:
            prob += 0.2
        elif tenure_months[i] > 36:
            prob -= 0.15
            
        # Contract effect
        if contract_types[i] == 'Month-to-month':
            prob += 0.3
        elif contract_types[i] == '2-year':
            prob -= 0.2
            
        # Satisfaction effect
        if satisfaction_scores[i] < 4:
            prob += 0.5
        elif satisfaction_scores[i] > 8:
            prob -= 0.2
            
        # Support calls effect
        if support_calls[i] > 5:
            prob += 0.3
            
        # Digital engagement effect
        if digital_engagement[i] < 20:
            prob += 0.2
            
        # Price sensitivity (higher charges = higher churn for some segments)
        if monthly_charges[i] > 150 and satisfaction_scores[i] < 6:
            prob += 0.2
            
        prob = max(0, min(1, prob))  # Clip between 0 and 1
        churn_probability.append(prob)
        churned.append(1 if np.random.random() < prob else 0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'age': ages,
        'gender': genders,
        'region': regions,
        'tenure_months': tenure_months,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'avg_monthly_usage_kwh': avg_monthly_usage,
        'contract_type': contract_types,
        'payment_method': payment_methods,
        'support_calls_last_year': support_calls,
        'digital_engagement_score': digital_engagement,
        'satisfaction_score': satisfaction_scores,
        'churn_probability': churn_probability,
        'churned': churned
    })
    
    # Add some realistic data quality issues
    # Missing satisfaction scores for some customers
    missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
    df.loc[missing_indices, 'satisfaction_score'] = np.nan
    
    # Some customers have missing total charges (new customers)
    new_customer_indices = df[df['tenure_months'] < 2].index
    if len(new_customer_indices) > 0:
        missing_total = np.random.choice(new_customer_indices, 
                                       size=min(10, len(new_customer_indices)), 
                                       replace=False)
        df.loc[missing_total, 'total_charges'] = np.nan
    
    return df

def generate_additional_datasets():
    """Generate additional datasets for comprehensive analysis"""
    
    # Campaign performance data
    campaign_data = pd.DataFrame({
        'campaign_id': [f'CAMP_{i:03d}' for i in range(1, 21)],
        'campaign_name': [
            'Summer Loyalty Program', 'New Customer Welcome', 'Win-Back Offer',
            'Premium Service Upgrade', 'Green Energy Promotion', 'Smart Home Bundle',
            'Senior Citizen Discount', 'Family Package Deal', 'Business Special',
            'Referral Rewards', 'Holiday Season Offer', 'Early Bird Renewal',
            'Student Discount Program', 'Low Usage Incentive', 'High Value Retention',
            'Digital Adoption Bonus', 'Payment Method Switch', 'Contract Extension',
            'Satisfaction Survey Follow-up', 'Emergency Support Package'
        ],
        'target_segment': [
            'Loyal Customers', 'New Customers', 'At-Risk Customers',
            'High Value', 'Environmentally Conscious', 'Tech Enthusiasts',
            'Senior Citizens', 'Families', 'Business Customers',
            'Advocates', 'All Customers', 'Contract Expiring',
            'Students', 'Low Usage', 'High Value At-Risk',
            'Low Digital Engagement', 'Cash Payers', 'Month-to-Month',
            'Dissatisfied Customers', 'High Support Users'
        ],
        'customers_targeted': np.random.randint(100, 1500, 20),
        'customers_responded': lambda x: np.random.randint(10, x, 20),
        'cost_per_customer': np.round(np.random.uniform(5, 25, 20), 2),
        'revenue_impact_per_customer': np.round(np.random.uniform(15, 100, 20), 2),
        'campaign_duration_days': np.random.randint(14, 90, 20)
    })
    
    # Fix the customers_responded column
    campaign_data['customers_responded'] = [
        np.random.randint(int(targeted * 0.05), int(targeted * 0.35)) 
        for targeted in campaign_data['customers_targeted']
    ]
    
    campaign_data['response_rate'] = (
        campaign_data['customers_responded'] / campaign_data['customers_targeted'] * 100
    ).round(2)
    
    campaign_data['total_cost'] = (
        campaign_data['customers_targeted'] * campaign_data['cost_per_customer']
    ).round(2)
    
    campaign_data['total_revenue_impact'] = (
        campaign_data['customers_responded'] * campaign_data['revenue_impact_per_customer']
    ).round(2)
    
    campaign_data['roi'] = (
        (campaign_data['total_revenue_impact'] - campaign_data['total_cost']) / 
        campaign_data['total_cost'] * 100
    ).round(2)
    
    return campaign_data

def main():
    """Generate all datasets and save to CSV files"""
    
    print("Generating customer data...")
    customer_df = generate_customer_data(5000)
    
    print("Generating campaign data...")
    campaign_df = generate_additional_datasets()
    
    # Create directory if it doesn't exist
    import os
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Save datasets
    customer_df.to_csv('data/raw/customer_data.csv', index=False)
    campaign_df.to_csv('data/raw/campaign_performance.csv', index=False)
    
    # Generate data quality report
    print("\n=== DATA QUALITY REPORT ===")
    print(f"Total customers generated: {len(customer_df):,}")
    print(f"Churn rate: {customer_df['churned'].mean():.2%}")
    print(f"Missing satisfaction scores: {customer_df['satisfaction_score'].isna().sum()}")
    print(f"Missing total charges: {customer_df['total_charges'].isna().sum()}")
    
    print("\n=== CUSTOMER SEGMENTS ===")
    print("By Contract Type:")
    print(customer_df['contract_type'].value_counts())
    
    print("\nBy Region (Top 5):")
    print(customer_df['region'].value_counts().head())
    
    print("\n=== CHURN ANALYSIS ===")
    print("Churn by Contract Type:")
    churn_by_contract = customer_df.groupby('contract_type')['churned'].agg(['count', 'sum', 'mean'])
    churn_by_contract.columns = ['Total', 'Churned', 'Churn_Rate']
    churn_by_contract['Churn_Rate'] = (churn_by_contract['Churn_Rate'] * 100).round(2)
    print(churn_by_contract)
    
    print(f"\nCampaign data generated: {len(campaign_df)} campaigns")
    print(f"Average campaign ROI: {campaign_df['roi'].mean():.1f}%")
    
    print("\nFiles saved successfully!")
    print("- data/raw/customer_data.csv")
    print("- data/raw/campaign_performance.csv")

if __name__ == "__main__":
    main()
