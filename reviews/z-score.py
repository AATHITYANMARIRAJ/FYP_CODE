import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Data for reasons (areas of difficulty)
reasons = [
    "Trigonometry", "Physics", "Trigonometry", "Physics", "Sequences and series", 
    "Rotational motion", "Chemistry", "Physics", "Trigonometry", "Physics"
]

# Count the frequency of each reason
reason_counts = Counter(reasons)

# Extract the reasons and their corresponding counts
reason_labels = list(reason_counts.keys())
reason_frequencies = list(reason_counts.values())

# Calculate the mean and standard deviation of the frequencies
mean_freq = np.mean(reason_frequencies)
std_freq = np.std(reason_frequencies)

# Calculate the z-scores for the reasons based on their frequencies
z_scores_reasons = [(x - mean_freq) / std_freq for x in reason_frequencies]

# Plotting the Z-scores for reasons
plt.figure(figsize=(8, 6))
plt.bar(reason_labels, z_scores_reasons, color='g')
plt.title('Z-scores of Reasons for Difficulty')
plt.xlabel('Reasons')
plt.ylabel('Z-score')
plt.grid(True)
plt.xticks(rotation=45, ha='right')
plt.show()

# Output Z-scores for each reason
for i, z in enumerate(z_scores_reasons):
    print(f"Reason: {reason_labels[i]}, Z-score: {z:.2f}")
