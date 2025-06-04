import matplotlib.pyplot as plt

# Define the ablation steps and the corresponding best accuracies for each dataset
study_names = ['Dropout', 'Depth', 'Hidden units', 'Learning rate', 'Weight decay']

# Replace these example values with your actual accuracy progressions if available
cora_accuracies =    [0.795, 0.810, 0.820, 0.825, 0.829]
citeseer_accuracies =[0.650, 0.670, 0.690, 0.695, 0.698]
pubmed_accuracies =  [0.680, 0.695, 0.700, 0.702, 0.705]

plt.figure(figsize=(8, 6))
plt.plot(study_names, cora_accuracies, marker='o', label='Cora', linewidth=2)
plt.plot(study_names, citeseer_accuracies, marker='s', label='Citeseer', linewidth=2)
plt.plot(study_names, pubmed_accuracies, marker='^', label='Pubmed', linewidth=2)

plt.xlabel('Ablation Study Step')
plt.ylabel('Best Test Accuracy')
plt.title('Ablation Study Progression: Accuracy per Tuning Step')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('result/stage_5_result/ablation_full/ablation_progression_all_datasets.png')
plt.show()
