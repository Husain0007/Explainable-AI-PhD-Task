import pandas as pd
import stellargraph as sg
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GCN
from sklearn.model_selection import StratifiedKFold
from keras import layers, optimizers, losses, Model
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import accuracy_score

# Load CORA dataset
cora_cites = pd.read_csv('cora.cites', sep="\t", header=None, names=["target", "source"])
cora_content = pd.read_csv('cora.content', sep="\t", header=None, names=["id", *["w"+str(i) for i in range(1433)], "subject"])

# Set index and split data
cora_content = cora_content.set_index("id")
cora_subject = cora_content["subject"]

# Prepare targets
target_encoding = preprocessing.LabelBinarizer()
cora_targets = target_encoding.fit_transform(cora_subject)

cora_content_no_subject = cora_content.drop(columns="subject")

cora_no_subject = sg.StellarGraph({"paper": cora_content_no_subject}, {"cites": cora_cites})

# Model configuration
generator = FullBatchNodeGenerator(cora_no_subject, method="gcn")

# Define the graph convolutional network model
gcn = GCN(layer_sizes=[16, 16], activations=["relu", "relu"], generator=generator, dropout=0.5)
x_inp, x_out = gcn.in_out_tensors()
predictions = layers.Dense(units=cora_targets.shape[1], activation="softmax")(x_out)
model = Model(inputs=x_inp, outputs=predictions)
model.compile(optimizer=optimizers.Adam(lr=0.01), loss=losses.categorical_crossentropy, metrics=["acc"])

# Instantiate the cross-validator
folds = 10
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

accuracies = []

# Loop through the epochs
for epoch in range(200):
    print(f"Epoch {epoch + 1}/200")
    
    # Get the current fold index
    fold_idx = epoch % folds
    
    # Get the train and validation indices for the current fold
    train_indices, val_indices = next(skf.split(cora_content_no_subject, cora_subject))

    print(f"Training and evaluating on fold {fold_idx+1} out of {folds}...")

    train_subject = cora_subject.iloc[train_indices]
    val_subject = cora_subject.iloc[val_indices]

    train_target = target_encoding.transform(train_subject)
    val_target = target_encoding.transform(val_subject)

    # Create train and validation generators
    train_gen = generator.flow(train_subject.index, train_target)
    val_gen = generator.flow(val_subject.index, val_target)

    # Train the model for this fold
    history = model.fit(train_gen, epochs=1, validation_data=val_gen, verbose=2)

    # Evaluate the model on the validation data
    val_predictions = model.predict(val_gen)
    # val_predictions_labels = np.argmax(val_predictions, axis=1)
    val_predictions = val_predictions.squeeze()
    val_predictions_labels = np.argmax(val_predictions, axis=1)
    val_true_labels = np.argmax(val_target, axis=1)
    val_accuracy = accuracy_score(val_true_labels, val_predictions_labels)
    accuracies.append(val_accuracy)    

print("Mean validation accuracy: ", np.mean(accuracies))

# Get predictions for all papers
all_gen = generator.flow(cora_content.index)
all_predictions = model.predict(all_gen)

# Inverse transform converts back the Softmax output to the label values
node_predictions = target_encoding.inverse_transform(all_predictions.squeeze())

# Ensure that the number of predictions matches the length of cora_content.index
if len(node_predictions) != len(cora_content.index):
    raise ValueError("Number of predictions does not match the length of cora_content.index")

# Store predictions in a DataFrame
predictions_df = pd.DataFrame({"paper_id": cora_content.index, "class_label": node_predictions})

# Save predictions to file
predictions_df.to_csv("predictions.tsv", sep="\t", index=False)

print("Overall accuracy on all predictions: ", accuracy_score(cora_subject, node_predictions))

print("Saving the final model as final_model.bin")
model.save("final_model.bin")