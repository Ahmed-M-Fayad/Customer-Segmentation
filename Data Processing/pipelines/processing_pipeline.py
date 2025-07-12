import pandas as pd


def process_customer_data(csv_path):
    """
    Process customer segmentation data for analysis.

    Args:
        csv_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Processed dataframe ready for segmentation
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

    # Feature engineering
    df["number_of_progeny"] = df["Kidhome"] + df["Teenhome"]
    df["years_since_joining_company"] = 2025 - df["customer_joining_year"]
    df["MntProds"] = (
        df["MntWines"]
        + df["MntFruits"]
        + df["MntMeatProducts"]
        + df["MntFishProducts"]
        + df["MntSweetProducts"]
        + df["MntGoldProds"]
    )
    df["AcceptedCmp"] = (
        df["AcceptedCmp3"]
        + df["AcceptedCmp4"]
        + df["AcceptedCmp5"]
        + df["AcceptedCmp1"]
        + df["AcceptedCmp2"]
    )

    # Encode education
    education_mapping = {"PhD": 3, "Master": 2, "Graduation": 1}
    df["Education"] = df["Education"].map(education_mapping)

    # One-hot encode marital status
    df = pd.get_dummies(df, columns=["Marital_Status"])

    # Drop processed columns
    columns_to_drop = [
        "ID",
        "AcceptedCmp3",
        "AcceptedCmp4",
        "AcceptedCmp5",
        "AcceptedCmp1",
        "AcceptedCmp2",
        "customer_joining_year",
        "Kidhome",
        "Teenhome",
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
    ]
    df = df.drop(columns=columns_to_drop)

    return df
