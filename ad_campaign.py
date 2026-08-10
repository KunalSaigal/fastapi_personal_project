import pandas as pd
import numpy as np

# Simulating marketing data for 6 different ad runs
data = {
    "Platform": ["Facebook", "Google", "Facebook", "Instagram", "Google", "Instagram"],
    "Ad_Clicks": [150, 400, 250, 300, 500, 100],
    "Revenue_USD": [300, 800, 550, 450, 1100, 150],
}

df = pd.DataFrame(data)
print("--- Raw Marketing Data ---")
print(df)

# Group by Platform, then sum up the numeric columns for each platform
platform_summary = df.groupby("Platform").sum()

print("\n--- Aggregated Platform Summary ---")
print(platform_summary)

# Custom aggregations per column
custom_summary = df.groupby("Platform").agg(
    {
        "Ad_Clicks": "mean",  # Get the average clicks per ad run
        "Revenue_USD": "sum",  # Get the total revenue generated
    }
)

# Renaming columns so they make sense
custom_summary = custom_summary.rename(
    columns={"Ad_Clicks": "Avg_Clicks", "Revenue_USD": "Total_Revenue"}
)

print("\n--- Custom Aggregation Summary ---")
print(custom_summary)
