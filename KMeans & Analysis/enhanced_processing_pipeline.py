import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def process_customer_data_enhanced(csv_path):
    """
    Enhanced customer segmentation data processing with comprehensive feature engineering.
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Processed dataframe with extensive feature engineering
    """
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Extract joining year
    df["customer_joining_year"] = df["Dt_Customer"].apply(
        lambda x: int(x.split("-")[2])
    )
    
    # Standardize marital status
    df["Marital_Status"] = df["Marital_Status"].replace("Together", "Married")
    df["Marital_Status"] = df["Marital_Status"].replace(
        ["Absurd", "YOLO", "Alone"], "Single"
    )
    
    # Standardize education
    df["Education"] = df["Education"].replace("2n Cycle", "Master")
    df["Education"] = df["Education"].replace("Basic", "Graduation")
    
    # Remove missing values and unnecessary columns
    df = df.dropna()
    df = df.drop(columns=["Z_CostContact", "Z_Revenue"])
    
    # Basic feature engineering (existing)
    df["number_of_progeny"] = df["Kidhome"] + df["Teenhome"]
    df["years_since_joining_company"] = 2025 - df["customer_joining_year"]
    df["MntProds"] = (
        df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] + 
        df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
    )
    df["AcceptedCmp"] = (
        df["AcceptedCmp3"] + df["AcceptedCmp4"] + df["AcceptedCmp5"] + 
        df["AcceptedCmp1"] + df["AcceptedCmp2"]
    )
    df['NumAllPurchases'] = (
        df['NumDealsPurchases'] + df['NumWebPurchases'] + 
        df['NumCatalogPurchases'] + df['NumStorePurchases']
    )
    
    # =============================================================================
    # ENHANCED FEATURE ENGINEERING
    # =============================================================================
    
    # 1. DEMOGRAPHIC & LIFECYCLE FEATURES
    # ------------------------------------
    
    # Age-related features
    df['Age'] = 2025 - df['Year_Birth']
    df['Age_Group'] = pd.cut(df['Age'], 
                            bins=[0, 35, 55, 100], 
                            labels=['Young', 'Middle', 'Senior'])
    
    # Generation mapping
    def get_generation(birth_year):
        if birth_year <= 1945:
            return 'Silent'
        elif birth_year <= 1964:
            return 'Baby_Boomer'
        elif birth_year <= 1980:
            return 'Gen_X'
        elif birth_year <= 1996:
            return 'Millennial'
        else:
            return 'Gen_Z'
    
    df['Generation'] = df['Year_Birth'].apply(get_generation)
    
    # Family structure features
    df['Total_Children'] = df['Kidhome'] + df['Teenhome']
    df['Family_Size'] = df['Total_Children'] + np.where(df['Marital_Status'] == 'Married', 2, 1)
    df['Has_Children'] = (df['Total_Children'] > 0).astype(int)
    df['Has_Teenagers'] = (df['Teenhome'] > 0).astype(int)
    df['Has_Young_Kids'] = (df['Kidhome'] > 0).astype(int)
    df['Child_Dependency_Ratio'] = df['Total_Children'] / df['Family_Size']
    
    # 2. FINANCIAL & SPENDING FEATURES
    # ---------------------------------
    
    # Income analysis
    df['Income_Per_Person'] = df['Income'] / df['Family_Size']
    df['Income_Quintile'] = pd.qcut(df['Income'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    df['Is_High_Income'] = (df['Income'] > df['Income'].quantile(0.75)).astype(int)
    
    # Spending patterns
    df['Total_Spending'] = df['MntProds']  # Already calculated
    df['Spending_Per_Income'] = df['Total_Spending'] / (df['Income'] + 1)  # +1 to avoid division by zero
    df['Spending_Per_Person'] = df['Total_Spending'] / df['Family_Size']
    df['Avg_Monthly_Spending'] = df['Total_Spending'] / (df['years_since_joining_company'] * 12 + 1)
    
    # Product category preferences (as ratios)
    df['Wine_Preference'] = df['MntWines'] / (df['Total_Spending'] + 1)
    df['Fruit_Preference'] = df['MntFruits'] / (df['Total_Spending'] + 1)
    df['Meat_Preference'] = df['MntMeatProducts'] / (df['Total_Spending'] + 1)
    df['Fish_Preference'] = df['MntFishProducts'] / (df['Total_Spending'] + 1)
    df['Sweet_Preference'] = df['MntSweetProducts'] / (df['Total_Spending'] + 1)
    df['Gold_Preference'] = df['MntGoldProds'] / (df['Total_Spending'] + 1)
    
    # Luxury vs Essential spending
    df['Luxury_Spending'] = df['MntWines'] + df['MntGoldProds']
    df['Essential_Spending'] = df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts']
    df['Luxury_Ratio'] = df['Luxury_Spending'] / (df['Total_Spending'] + 1)
    df['Essential_Ratio'] = df['Essential_Spending'] / (df['Total_Spending'] + 1)
    
    # 3. PURCHASE BEHAVIOR FEATURES
    # ------------------------------
    
    # Channel preferences
    df['Web_Purchase_Ratio'] = df['NumWebPurchases'] / (df['NumAllPurchases'] + 1)
    df['Store_Purchase_Ratio'] = df['NumStorePurchases'] / (df['NumAllPurchases'] + 1)
    df['Catalog_Purchase_Ratio'] = df['NumCatalogPurchases'] / (df['NumAllPurchases'] + 1)
    df['Deal_Purchase_Ratio'] = df['NumDealsPurchases'] / (df['NumAllPurchases'] + 1)
    
    # Multi-channel usage
    channels = ['NumWebPurchases', 'NumStorePurchases', 'NumCatalogPurchases']
    df['Channels_Used'] = sum((df[col] > 0).astype(int) for col in channels)
    df['Is_Multi_Channel'] = (df['Channels_Used'] > 1).astype(int)
    
    # Preferred channel
    channel_cols = ['NumWebPurchases', 'NumStorePurchases', 'NumCatalogPurchases']
    df['Preferred_Channel'] = df[channel_cols].idxmax(axis=1)
    
    # Deal sensitivity
    df['Deal_Sensitivity'] = df['NumDealsPurchases'] / (df['NumAllPurchases'] + 1)
    df['Is_Deal_Hunter'] = (df['Deal_Sensitivity'] > df['Deal_Sensitivity'].median()).astype(int)
    
    # Purchase frequency
    df['Avg_Purchase_Frequency'] = df['NumAllPurchases'] / (df['years_since_joining_company'] + 1)
    df['Days_Between_Purchases'] = (365 * df['years_since_joining_company']) / (df['NumAllPurchases'] + 1)
    
    # Spending per purchase
    df['Avg_Spending_Per_Purchase'] = df['Total_Spending'] / (df['NumAllPurchases'] + 1)
    
    # 4. ENGAGEMENT & CAMPAIGN FEATURES
    # ----------------------------------
    
    # Campaign response
    df['Campaign_Response_Rate'] = df['AcceptedCmp'] / 5  # 5 total campaigns
    df['Is_Campaign_Responsive'] = (df['AcceptedCmp'] > 0).astype(int)
    
    # Web engagement
    df['Web_Visits_Per_Month'] = df['NumWebVisitsMonth']
    df['Web_Visits_Per_Purchase'] = df['NumWebVisitsMonth'] / (df['NumWebPurchases'] + 1)
    df['Web_Conversion_Rate'] = df['NumWebPurchases'] / (df['NumWebVisitsMonth'] + 1)
    df['Is_Web_Browser'] = (df['NumWebVisitsMonth'] > df['NumWebVisitsMonth'].median()).astype(int)
    
    # Customer satisfaction proxy
    df['Satisfaction_Score'] = (1 - df['Complain']) * df['Campaign_Response_Rate']
    
    # 5. RECENCY & LOYALTY FEATURES
    # ------------------------------
    
    # Recency groups
    df['Recency_Group'] = pd.cut(df['Recency'], 
                                bins=[0, 30, 90, float('inf')], 
                                labels=['Active', 'Moderate', 'Inactive'])
    df['Is_Recent_Customer'] = (df['Recency'] <= 30).astype(int)
    
    # Loyalty metrics
    df['Customer_Lifetime_Value'] = df['Total_Spending']
    df['Annual_Value'] = df['Total_Spending'] / (df['years_since_joining_company'] + 1)
    df['Loyalty_Score'] = (df['years_since_joining_company'] * 
                          (1 - df['Complain']) * 
                          df['Campaign_Response_Rate'])
    
    # 6. INTERACTION & RATIO FEATURES
    # --------------------------------
    # Encode education
    education_mapping = {"PhD": 3, "Master": 2, "Graduation": 1}
    df["Education_encoded"] = df["Education"].map(education_mapping)
    
    # Cross-feature interactions
    df['Income_Age_Interaction'] = df['Income'] * df['Age']
    df['Family_Income_Interaction'] = df['Family_Size'] * df['Income']
    df['Spending_Age_Interaction'] = df['Total_Spending'] * df['Age']
    df['Income_Education_Interaction'] = df['Income'] * df['Education_encoded']
    
    # Efficiency ratios
    df['Purchases_Per_Visit'] = df['NumWebPurchases'] / (df['NumWebVisitsMonth'] + 1)
    df['Spending_Per_Visit'] = df['Total_Spending'] / (df['NumWebVisitsMonth'] + 1)
    
    # 7. BINNING & CATEGORICAL FEATURES
    # ----------------------------------
    
    # Spending tiers
    df['Spending_Tier'] = pd.qcut(df['Total_Spending'], 
                                 3, 
                                 labels=['Low', 'Medium', 'High'])
    
    # Income tiers
    df['Income_Tier'] = pd.qcut(df['Income'], 
                               3, 
                               labels=['Low', 'Medium', 'High'])
    
    # Customer lifecycle stage
    def get_life_stage(row):
        if row['Age'] < 35:
            if row['Marital_Status'] == 'Single':
                return 'Young_Single'
            elif row['Has_Children']:
                return 'Young_Family'
            else:
                return 'Young_Couple'
        elif row['Age'] < 55:
            if row['Has_Children']:
                return 'Middle_Family'
            else:
                return 'Middle_Couple'
        else:
            if row['Total_Children'] == 0:
                return 'Empty_Nesters'
            else:
                return 'Senior_Family'
    
    df['Life_Stage'] = df.apply(get_life_stage, axis=1)
    
    # 8. ADVANCED BEHAVIORAL SEGMENTS
    # --------------------------------
    
    # Customer type based on behavior
    def get_customer_type(row):
        if row['Web_Purchase_Ratio'] > 0.5 and row['Total_Spending'] > row['Total_Spending'].quantile(0.75):
            return 'High_Value_Web'
        elif row['Store_Purchase_Ratio'] > 0.5 and row['Total_Spending'] > row['Total_Spending'].quantile(0.75):
            return 'High_Value_Store'
        elif row['Deal_Sensitivity'] > 0.5:
            return 'Deal_Hunter'
        elif row['Campaign_Response_Rate'] > 0.4:
            return 'Campaign_Responsive'
        elif row['Luxury_Ratio'] > 0.5:
            return 'Luxury_Buyer'
        else:
            return 'Regular_Customer'
    
    # Apply customer type (need to calculate quantiles first)
    spending_q75 = df['Total_Spending'].quantile(0.75)
    def get_customer_type_fixed(row):
        if row['Web_Purchase_Ratio'] > 0.5 and row['Total_Spending'] > spending_q75:
            return 'High_Value_Web'
        elif row['Store_Purchase_Ratio'] > 0.5 and row['Total_Spending'] > spending_q75:
            return 'High_Value_Store'
        elif row['Deal_Sensitivity'] > 0.5:
            return 'Deal_Hunter'
        elif row['Campaign_Response_Rate'] > 0.4:
            return 'Campaign_Responsive'
        elif row['Luxury_Ratio'] > 0.5:
            return 'Luxury_Buyer'
        else:
            return 'Regular_Customer'
    
    df['Customer_Type'] = df.apply(get_customer_type_fixed, axis=1)
    
    # 9. POLYNOMIAL FEATURES (for key metrics)
    # -----------------------------------------
    
    # Squared terms for important features
    df['Income_Squared'] = df['Income'] ** 2
    df['Total_Spending_Squared'] = df['Total_Spending'] ** 2
    df['Age_Squared'] = df['Age'] ** 2
    df['Recency_Squared'] = df['Recency'] ** 2
    
    # 10. ENCODE CATEGORICAL VARIABLES
    # ---------------------------------
    
    # One-hot encode marital status
    df = pd.get_dummies(df, columns=["Marital_Status"], prefix="Marital_Status")
    
    # One-hot encode new categorical features
    categorical_features = ['Age_Group', 'Generation', 'Income_Quintile', 
                           'Spending_Tier', 'Income_Tier', 'Life_Stage', 
                           'Customer_Type', 'Recency_Group', 'Preferred_Channel']
    
    for feature in categorical_features:
        if feature in df.columns:
            df = pd.get_dummies(df, columns=[feature], prefix=feature)
    
    # Convert boolean columns to int
    bool_columns = [col for col in df.columns if col.startswith('Marital_Status_') or 
                   col.startswith('Age_Group_') or col.startswith('Generation_') or
                   col.startswith('Income_Quintile_') or col.startswith('Spending_Tier_') or
                   col.startswith('Income_Tier_') or col.startswith('Life_Stage_') or
                   col.startswith('Customer_Type_') or col.startswith('Recency_Group_') or
                   col.startswith('Preferred_Channel_')]
    
    for col in bool_columns:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    # Drop original columns that were encoded
    columns_to_drop = [
        "ID", "Education", "Dt_Customer", "customer_joining_year"
    ]
    
    # Only drop columns that exist
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=columns_to_drop)
    
    # Fill any remaining NaN values
    df = df.fillna(0)
    
    return df

# Updated main processing function for your existing pipeline
def process_customer_data(csv_path):
    """
    Original processing function - kept for backward compatibility
    """
    return process_customer_data_enhanced(csv_path)

# Function to prepare data for clustering (similar to your existing approach)
def prepare_clustering_data(df):
    """
    Prepare the feature-engineered data for clustering
    
    Args:
        df: DataFrame from process_customer_data_enhanced
        
    Returns:
        tuple: (scaled_data, categorical_data, feature_names)
    """
    
    # Identify categorical columns (binary indicators)
    cat_cols = [col for col in df.columns if 
                col.startswith('Marital_Status_') or 
                col.startswith('Age_Group_') or 
                col.startswith('Generation_') or
                col.startswith('Income_Quintile_') or
                col.startswith('Spending_Tier_') or
                col.startswith('Income_Tier_') or
                col.startswith('Life_Stage_') or
                col.startswith('Customer_Type_') or
                col.startswith('Recency_Group_') or
                col.startswith('Preferred_Channel_') or
                col in ['Complain', 'Response', 'AcceptedCmp3', 'AcceptedCmp4', 
                       'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Has_Children',
                       'Has_Teenagers', 'Has_Young_Kids', 'Is_High_Income', 
                       'Is_Multi_Channel', 'Is_Deal_Hunter', 'Is_Campaign_Responsive',
                       'Is_Web_Browser', 'Is_Recent_Customer']]
    
    # Separate categorical and numerical data
    cat_data = df[cat_cols]
    num_data = df.drop(columns=cat_cols)
    
    # Scale numerical data
    scaler = StandardScaler()
    scaled_num_data = scaler.fit_transform(num_data)
    
    # Combine scaled numerical and categorical data
    all_data = pd.concat([pd.DataFrame(scaled_num_data, columns=num_data.columns), 
                         cat_data.reset_index(drop=True)], axis=1)
    
    # Fill any remaining NaN values
    all_data = all_data.fillna(0)
    
    return all_data, cat_data, num_data.columns.tolist()

# Example usage function
def run_enhanced_pipeline(csv_path):
    """
    Run the complete enhanced pipeline
    """
    print("Processing customer data with enhanced feature engineering...")
    
    # Process data with enhanced features
    data = process_customer_data_enhanced(csv_path)
    
    print(f"Dataset shape after feature engineering: {data.shape}")
    print(f"Number of features: {len(data.columns)}")
    
    # Prepare for clustering
    all_data, cat_data, num_features = prepare_clustering_data(data)
    
    print(f"Features for clustering: {all_data.shape[1]}")
    print(f"Categorical features: {len(cat_data.columns)}")
    print(f"Numerical features: {len(num_features)}")
    
    return data, all_data, cat_data, num_features