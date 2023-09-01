import matplotlib.pyplot as plt

# List of method names
methods = ["VADER", "TextBlob", "SentiWordNet", "Affective Norms", "CNN"]

# Corresponding time elapsed in seconds
time_elapsed = [316.48, 149.00, 1613.69, 12.33, 510.12]

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(methods, time_elapsed, color=['blue', 'green', 'red', 'purple', 'orange'])
plt.xlabel("Sentiment Analysis Method")
plt.ylabel("Time Elapsed (s)")
plt.title("Time Elapsed for Sentiment Analysis Methods")
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()
