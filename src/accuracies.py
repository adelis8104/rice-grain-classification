import matplotlib.pyplot as plt

# Example accuracies (replace with actual values if needed)
#models = ['KNN Base', 'KNN minkowski', 'KNN minkowski 20%']
#accuracies = [0.9775, 0.9780, 0.971]  # Placeholder values
#models = ['CNN Base', 'CNN 20%', 'CNN 10%','CNN 128 batch 10%', 'CNN 128 batch 20%', 'CNN 128 batch 20 epoch 7']
#accuracies = [0.9695, 0.9873, 0.9780, 0.9800, 0.9560, 0.9847]  # Placeholder values

models = ['CNN Base', 'CNN 20%', 'CNN 10%','CNN 128 batch 10%', 'CNN 128 batch 20%', 'CNN 128 batch 20 epoch 7']
accuracies = [0.9695, 0.9873, 0.9780, 0.9800, 0.9560, 0.9847]  # Placeholder values

# Plot
plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracies, color='skyblue')
plt.ylim(0.9, 1.0)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Classification Method')

# Annotate bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.4f}', ha='center', va='bottom')

# Save plot
plot_path = "C:/Users/Antonio/Documents/CS 659 Project/rice-grain-classification/Results/model_accuracy_comparison.png"
plt.tight_layout()
plt.savefig(plot_path)
plt.close()

plot_path