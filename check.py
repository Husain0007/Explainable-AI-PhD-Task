import pandas as pd
import stellargraph as sg
from stellargraph.mapper import FullBatchNodeGenerator
from keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


# Load CORA dataset
cora_cites = pd.read_csv('./cora/cora.cites', sep="\t", header=None, names=["target", "source"])
cora_content = pd.read_csv('./cora/cora.content', sep="\t", header=None, names=["id", *["w"+str(i) for i in range(1433)], "subject"])

# Set index and split data
cora_content = cora_content.set_index("id")
cora_subject = cora_content["subject"]

# Prepare targets
target_encoding = preprocessing.LabelBinarizer()
cora_targets = target_encoding.fit_transform(cora_subject)

cora_content_no_subject = cora_content.drop(columns="subject")

cora_no_subject = sg.StellarGraph({"paper": cora_content_no_subject}, {"cites": cora_cites})

# Load the saved model
model = load_model("final_model.bin")

# Create a generator for all papers
generator = FullBatchNodeGenerator(cora_no_subject, method="gcn")
all_gen = generator.flow(cora_content.index)

# Perform inference
all_predictions = model.predict(all_gen)
node_predictions = target_encoding.inverse_transform(all_predictions.squeeze())

# Calculate overall accuracy
overall_accuracy = accuracy_score(cora_subject, node_predictions)
print("Overall accuracy on all predictions: ", overall_accuracy*100)

# Store predictions in a DataFrame
predictions_df = pd.DataFrame({"paper_id": cora_content.index, "class_label": node_predictions})

# Save predictions to file
predictions_df.to_csv("inference_predictions.tsv", sep="\t", index=False)

# Visualize recall for each class
class_report = classification_report(cora_subject, node_predictions, output_dict=True)
class_names = target_encoding.classes_
recall_values = [class_report[label]['recall'] for label in class_names]
precision_values = [class_report[label]['precision'] for label in class_names]

plt.figure(figsize=(15, 10))

# Define colors
correct_color = 'lightgreen'
incorrect_color = 'lightcoral'

# Plot correct recall values in pastel green
bar_plot = plt.bar(class_names, recall_values, color=correct_color, label='Recall Success Rate')

# Plot incorrect recall values in pastel red next to the correct recall bars
for i, recall in enumerate(recall_values):
    plt.text(i, recall + 0.005, f'{recall*100:.2f}%', ha='center', va='bottom', color='black', fontweight='bold')
    plt.bar(i, 1 - recall, bottom=recall, color=incorrect_color, alpha=0.7)

# Display class names inside the bar plots
for bar, label in zip(bar_plot, class_names):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height / 2, label, ha='center', va='center', rotation='vertical', color='black', fontsize=15, fontweight='bold')

plt.xlabel('Class', fontsize=12, fontweight='bold')
plt.ylabel('Recall', fontsize=12, fontweight='bold')
plt.title('Recall for Each Class', fontsize=14, fontweight='bold')

# Update legend to include recall failure rate
plt.legend(handles=[bar_plot[0], plt.Rectangle((0,0),1,1,color=incorrect_color, alpha=0.7)], labels=['Recall Success Rate', 'Recall Failure Rate'])

# Remove x-axis labels
plt.xticks([])

# Save the figure as a .png image
plt.savefig('recall_bargraph.png', bbox_inches='tight')
plt.close()