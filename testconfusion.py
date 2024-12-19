import json
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from safetensors.torch import load_file
from tqdm import tqdm

# Paths to your model and associated files
model_path = "./model.safetensors"
tokenizer_path = "../news-classification-model"
config_path = "../news-classification-model"
special_tokens_path = "./special_token_map.json"

# Load the Tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Load Special Tokens (if applicable)
try:
    with open(special_tokens_path, "r") as f:
        special_tokens = json.load(f)
    tokenizer.add_special_tokens(special_tokens)
except FileNotFoundError:
    pass

# Load the Model
state_dict = load_file(model_path)
model = AutoModelForSequenceClassification.from_pretrained(
    config_path,
    state_dict=state_dict,
    torch_dtype=torch.float32
)
model.resize_token_embeddings(len(tokenizer))

# Preprocess Text
def preprocess_text(text, tokenizer, max_length=512):
    return tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

# Predict Function
def predict(text, model, tokenizer, device="cpu"):
    inputs = preprocess_text(text, tokenizer)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    model = model.to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()

    return predicted_class, probabilities

def predict_category(input_text, model, tokenizer, device, config_path):
    predicted_class, _ = predict(input_text, model, tokenizer, device)

    with open(f"{config_path}/config.json", "r") as f:
        config = json.load(f)
    class_names = config.get("id2label", {})

    predicted_class_name = class_names.get(str(predicted_class), "Unknown")
    return predicted_class_name

def create_test_dataset(df, k_samples_per_category):
    """
    Create a test dataset by sampling K samples from each category.
    
    Args:
        df (pd.DataFrame): Input dataframe with samples and labels
        k_samples_per_category (int): Number of samples to select from each category
    
    Returns:
        pd.DataFrame: Test dataset
        list: Text samples
        list: Original labels
    """
    # Group by category and sample K samples from each
    test_samples = df.groupby('target').apply(
        lambda x: x.sample(n=min(k_samples_per_category, len(x)))
    ).reset_index(drop=True)
    
    # Separate text and labels
    test_texts = test_samples['processed_text'].tolist()
    true_labels = test_samples['target'].tolist()
    
    return test_samples, test_texts, true_labels

def plot_confusion_matrix(true_labels, predicted_labels, categories):
    """
    Plot and save confusion matrix.
    
    Args:
        true_labels (list): True category labels
        predicted_labels (list): Predicted category labels
        categories (list): Unique categories
    """
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=categories)
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=categories, 
                yticklabels=categories)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix-2.png')
    plt.close()

def generate_performance_report(true_labels, predicted_labels):
    """
    Generate and save performance metrics.
    
    Args:
        true_labels (list): True category labels
        predicted_labels (list): Predicted category labels
    
    Returns:
        dict: Performance metrics
    """
    # Compute precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
    
    # Generate classification report
    report = classification_report(true_labels, predicted_labels, output_dict=True)
    
    # Prepare performance metrics
    performance_metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'detailed_report': report
    }
    
    # Save detailed report to JSON
    with open('performance_report-2.json', 'w') as f:
        json.dump(performance_metrics, f, indent=4)
    
    return performance_metrics

def plot_performance_metrics(metrics):
    """
    Plot performance metrics.
    
    Args:
        metrics (dict): Performance metrics dictionary
    """
    # Exclude summary rows
    categories = [cat for cat in metrics['detailed_report'].keys() 
                  if cat not in ['accuracy', 'macro avg', 'weighted avg']]
    
    # Prepare data for plotting
    precisions = [metrics['detailed_report'][cat]['precision'] for cat in categories]
    recalls = [metrics['detailed_report'][cat]['recall'] for cat in categories]
    f1_scores = [metrics['detailed_report'][cat]['f1-score'] for cat in categories]
    accuracies = [metrics['detailed_report'][cat]['support'] / sum(metrics['detailed_report'][cat]['support'] for cat in categories) for cat in categories]
    
    # Plot
    plt.figure(figsize=(14, 7))
    x = np.arange(len(categories))
    width = 0.25
    
    plt.bar(x - width, precisions, width, label='Precision', color='blue', alpha=0.7)
    plt.bar(x, recalls, width, label='Recall', color='green', alpha=0.7)
    plt.bar(x + width, f1_scores, width, label='F1-Score', color='red', alpha=0.7)
    
    plt.xlabel('Categories', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Performance Metrics by Category', fontsize=16)
    plt.xticks(x, categories, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('performance_metrics-2.png')
    plt.close()

def main():
    # Specify device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Input and output file paths
    input_csv_path = "combined_output.csv"  # CSV with labeled samples
    output_file_path = "predicted_results_1.json"
    
    # Hyperparameters
    k_samples_per_category = 100  # Number of samples to select from each category for testing

    # Read input samples from CSV
    df = pd.read_csv(input_csv_path,encoding='latin1')
    print(df.head())
    
    # Create test dataset
    test_samples, test_texts, true_labels = create_test_dataset(df, k_samples_per_category)

    # Perform predictions
    predicted_labels = []
    results = []
  

    for input_text in tqdm(test_texts, desc="Processing texts"):
        predicted_category = predict_category(input_text, model, tokenizer, device, config_path)
        predicted_labels.append(predicted_category)
        results.append({
            "text": input_text, 
            "true_category": true_labels[test_texts.index(input_text)],
            "predicted_category": predicted_category
        })

    # Write predictions to output file
    with open(output_file_path, "w") as output_file:
        json.dump(results, output_file, indent=4)

    print(f"Predictions written to {output_file_path}")

    # Get unique categories
    categories = sorted(df['target'].unique())

    # Plot confusion matrix
    plot_confusion_matrix(true_labels, predicted_labels, categories)

    # Generate performance report
    performance_metrics = generate_performance_report(true_labels, predicted_labels)

    # Plot performance metrics
    plot_performance_metrics(performance_metrics)

    # Print overall performance
    print(f"Precision: {performance_metrics['precision']:.4f}")
    print(f"Recall: {performance_metrics['recall']:.4f}")
    print(f"F1-Score: {performance_metrics['f1_score']:.4f}")

if __name__ == "__main__":
    main()