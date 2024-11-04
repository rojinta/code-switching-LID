from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
from transformers import get_linear_schedule_with_warmup
import torch
from torch.nn.utils import clip_grad_norm_
from CS_Dataset import CSDataset
from loss import HybridLoss

from sklearn.metrics import f1_score, classification_report
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import logging
import os
import random
from datetime import datetime
import json


def setup_logging(log_dir="logs"):
    """Set up logging configuration"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def set_seed(seed=2731):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_config(config, save_dir):
    """Save training configuration"""
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


def compute_class_weights(dataset):
    """calculate class weights for imbalanced datasets"""
    all_labels = []
    for batch in DataLoader(dataset, batch_size=32):
        labels = batch['labels'].view(-1)  # flatten
        valid_labels = [l.item() for l in labels if l.item() != -100]
        all_labels.extend(valid_labels)

    # calculate class weights
    class_counts = np.bincount(all_labels)

    beta = 0.999
    weights = (1 - beta) / (1 - beta ** class_counts)
    # or use the following
    # weights = np.sqrt(1 / class_counts)
    weights = weights / weights.sum() * len(weights)

    return torch.FloatTensor(weights)


def compute_metrics(pred_labels, true_labels, attention_mask):
    # only consider the active parts of the sequence
    active_labels = true_labels[attention_mask == 1]
    active_preds = pred_labels[attention_mask == 1]

    f1_macro = f1_score(active_labels.cpu(), active_preds.cpu(), average='macro', zero_division=0)
    f1_weighted = f1_score(active_labels.cpu(), active_preds.cpu(), average='weighted', zero_division=0)

    # calculate classification report
    report = classification_report(
        active_labels.cpu(),
        active_preds.cpu(),
        output_dict=True,
        zero_division=0
    )

    return {
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'classification_report': report
    }


def evaluate_model(model, test_dataset, device):
    model.eval()
    dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_predictions = []
    all_labels = []
    all_masks = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(inputs, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)

            all_predictions.append(predictions)
            all_labels.append(labels)
            all_masks.append(attention_mask)

    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    metrics = compute_metrics(all_predictions, all_labels, all_masks)

    return metrics


def train(config):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """Main training function"""
    logger = setup_logging()
    set_seed(config['seed'])

    # Save configuration
    save_dir = os.path.join("outputs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    save_config(config, save_dir)

    pretrained_model = config['model_name']
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)


    train_dataset = CSDataset('lid_spaeng/train.conll', tokenizer, mask_out_prob=0.25)
    eval_dataset = CSDataset('lid_spaeng/dev.conll', tokenizer, mask_out_prob=0)
    test_dataset = CSDataset('lid_hineng/dev.conll', tokenizer, mask_out_prob=0)

    label2id = train_dataset.label2id
    id2label = train_dataset.id2label
    label_list = list(label2id.keys())

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    model = AutoModelForTokenClassification.from_pretrained(
        pretrained_model, num_labels=len(label_list), id2label=id2label, label2id=label2id
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    # add learning rate scheduler
    num_training_steps = len(train_dataloader) * config['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_training_steps // config['warmup_ratio'],
        num_training_steps=num_training_steps
    )

    class_weights = compute_class_weights(train_dataset).to(device)
    # loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    loss_fn = HybridLoss(weights=class_weights, alpha=0.5, gamma=2.0)

    train_losses = []
    dev_f1_macro = []
    dev_f1_weighted = []
    best_f1 = 0

    model.train()
    logger.info("Starting training...")
    for epoch in range(config['num_epochs']):
        epoch_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}"):
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(inputs, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits.view(-1, model.config.num_labels), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            scheduler.step()
            optimizer.step()

            epoch_loss += loss.item()

        average_loss = epoch_loss / len(train_dataloader)
        train_losses.append(average_loss)

        # evaluate on the dev set
        dev_metrics = evaluate_model(model, eval_dataset, device)
        dev_f1_macro.append(dev_metrics['f1_macro'])  # F1 Macro
        dev_f1_weighted.append(dev_metrics['f1_weighted'])  # F1 Weighted
        print(
            f"Epoch {epoch + 1}/{config['num_epochs']}, Train Loss: {average_loss:.4f}, Dev F1 Macro: {dev_metrics['f1_macro']:.4f}, "
            f"Dev F1 Weighted: {dev_metrics['f1_weighted']:.4f}")

        logger.info(
            f"Epoch {epoch + 1}/{config['num_epochs']}: "
            f"Train Loss = {average_loss:.4f}, "
            f"Dev F1 Macro = {dev_metrics['f1_macro']:.4f}, "
            f"Dev F1 Weighted = {dev_metrics['f1_weighted']:.4f}"
        )

        # Save best model
        if dev_metrics['f1_macro'] > best_f1:
            best_f1 = dev_metrics['f1_macro']
            model_save_path = os.path.join(save_dir, "best_model")
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            logger.info(f"Saved new best model with F1 Macro: {best_f1:.4f}")

    logger.info("Evaluating on test set...")
    test_metrics = evaluate_model(model, test_dataset, device)
    logger.info(f"Test Results: F1 Macro = {test_metrics['f1_macro']:.4f}, "
                f"F1 Weighted = {test_metrics['f1_weighted']:.4f}")
    print(f"Test F1 Macro: {test_metrics['f1_macro']:.4f}, Test F1 Weighted: {test_metrics['f1_weighted']:.4f}")

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Curve')

    plt.subplot(1, 2, 2)
    plt.plot(dev_f1_macro, label='F1 Macro')
    plt.plot(dev_f1_weighted, label='F1 Weighted')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Scores')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"))
    plt.close()

    return {
        'train_losses': train_losses,
        'dev_f1_macro': dev_f1_macro,
        'dev_f1_weighted': dev_f1_weighted,
        'test_metrics': test_metrics,
        'best_f1': best_f1
    }


if __name__ == "__main__":
    config = {
        'model_name': "bert-base-multilingual-cased",
        'train_file': 'lid_spaeng/train.conll',
        'eval_file': 'lid_spaeng/dev.conll',
        'test_file': 'lid_hineng/train.conll',
        'batch_size': 128,
        'learning_rate': 5e-5,
        'weight_decay': 0.01,
        'num_epochs': 10,
        'warmup_ratio': 10,
        'max_grad_norm': 1.0,
        'seed': 2137
    }

    results = train(config)




