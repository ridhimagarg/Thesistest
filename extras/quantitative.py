import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_context("paper", font_scale=2.0, rc={"lines.linewidth": 1.5})

# Sample data
data = {
    'Category': ['FS', 'FS', 'XFS', 'XFS', 'XFS', 'EWE-R', 'EWE-R', "EWE-F" ,"EWE-F", "DAWN"],
    'Model': ["MN", "CF-L", "MN", "CF-L", "CF-H", "MN", "CF-L", "MN", "CF-L", ""],
    'Value': [58.49, 147.88 ,19.87, 43.72 ,102.83, 125.23 , 310.78 , 123.04, 299.77, 0]
}

df = pd.DataFrame(data)

palette = {"MN": "blue", "CF-L": "tab:orange", "CF-H":"red", "": "red"}

# Create a grouped horizontal bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Value', y='Category', hue='Model', data=df,  palette=palette)

# Add labels and title
plt.ylabel('Watermarking Technique')
plt.xlabel('Embedding Time (in seconds)')
plt.title('Watermark Embedding Times')

# Display the plot
plt.show()
plt.savefig("time.svg", bbox_inches='tight')