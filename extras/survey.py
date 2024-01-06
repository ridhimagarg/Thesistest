import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# As the image input cannot be directly read into a dataframe, we will create the data manually
# Let's assume the values from the graph provided by the user (this is an approximation)

# Sample data
data = {
    'Year': [2017, 2018, 2019, 2020, 2021, 2022, "Sept-Oct-2023"],
    'Watermarking': [1, 3, 7, 11, 8, 6 , 3] , # These values are placeholders
    'Watermarking for ME': [0, 0, 0, 0, 2, 1, 2]  # These values are placeholders
}

# Convert to DataFrame
df = pd.DataFrame(data)

# We need to melt the dataframe to have proper format for seaborn
df_melted = df.melt('Year', var_name='Watermarking', value_name='Amount of publications')

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Year', y='Amount of publications', hue='Watermarking', data=df_melted)

plt.title('Amount of Publications Over Years')
plt.tight_layout()
plt.show()
plt.savefig("watermarking.svg", bbox_inches='tight')
