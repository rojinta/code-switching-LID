from transformers import BertForTokenClassification, Trainer, TrainingArguments
from transformers import get_linear_schedule_with_warmup
import torch
from CS_Dataset import CSDataset

from sklearn.metrics import f1_score, classification_report
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


def compute_class_weights(dataset):
    """calculate class weights for imbalanced datasets"""
    all_labels = []
    for batch in DataLoader(dataset, batch_size=32):
        labels = batch['labels'].view(-1)  # flatten
        all_labels.extend(labels.cpu().numpy())

    # calculate class weights
    class_counts = np.bincount(all_labels)
    weights = 1 / class_counts
    weights = weights / weights.min()  # 归一化

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
            inputs = batch['inputs_embeds'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(inputs_embeds=inputs, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)

            all_predictions.append(predictions)
            all_labels.append(labels)
            all_masks.append(attention_mask)

    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    metrics = compute_metrics(all_predictions, all_labels, all_masks)

    return metrics


def train(num_epochs=200):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=8).to(device)

    train_dataset = CSDataset('train', 'en-es', mask_out_prob=0.4)
    eval_dataset = CSDataset('dev', 'en-es', mask_out_prob=0)
    test_dataset = CSDataset('dev', 'en-hi', mask_out_prob=0)

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    # add learning rate scheduler
    num_training_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_training_steps // 10,
        num_training_steps=num_training_steps
    )

    class_weights = compute_class_weights(train_dataset).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    train_losses = []
    dev_f1_macro = []
    dev_f1_weighted = []

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in tqdm(train_dataloader):
            inputs = batch['inputs_embeds'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(inputs_embeds=inputs, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits.view(-1, model.config.num_labels), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
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
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {average_loss:.4f}, Dev F1 Macro: {dev_metrics['f1_macro']:.4f}, Dev F1 Weighted: {dev_metrics['f1_weighted']:.4f}")

    test_metrics = evaluate_model(model, test_dataset, device)
    print(f"Test F1 Macro: {test_metrics['f1_macro']:.4f}, Test F1 Weighted: {test_metrics['f1_weighted']:.4f}")

    return train_losses, dev_f1_macro, dev_f1_weighted

    # training_args = TrainingArguments(
    #     output_dir='results',
    #     evaluation_strategy='epoch',
    #     learning_rate=2e-5,
    #     per_device_train_batch_size=64,
    #     per_device_eval_batch_size=64,
    #     num_train_epochs=1,
    #     logging_dir='logs',
    #     logging_steps=10,
    #     save_strategy="epoch"
    # )
    #
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    # )
    #
    # trainer.train()
    #
    # print("\nEvaluating on test set...")
    # test_metrics = evaluate_model(model, test_dataset, device)
    #
    # print("\nTest Set Results:")
    # print(f"Macro F1: {test_metrics['f1_macro']:.4f}")
    # print(f"Weighted F1: {test_metrics['f1_weighted']:.4f}")
    # print("\nDetailed Classification Report:")
    # for label, metrics in test_metrics['classification_report'].items():
    #     if isinstance(metrics, dict):
    #         print(f"\nClass {label}:")
    #         print(f"Precision: {metrics['precision']:.4f}")
    #         print(f"Recall: {metrics['recall']:.4f}")
    #         print(f"F1-score: {metrics['f1-score']:.4f}")
    #         print(f"Support: {metrics['support']}")
    #
    #     save_path = 'test_results.txt'
    #     with open(save_path, 'w') as f:
    #         f.write("Test Set Results:\n")
    #         f.write(f"Macro F1: {test_metrics['f1_macro']:.4f}\n")
    #         f.write(f"Weighted F1: {test_metrics['f1_weighted']:.4f}\n")
    #         f.write("\nDetailed Classification Report:\n")
    #         for label, metrics in test_metrics['classification_report'].items():
    #             if isinstance(metrics, dict):
    #                 f.write(f"\nClass {label}:\n")
    #                 f.write(f"Precision: {metrics['precision']:.4f}\n")
    #                 f.write(f"Recall: {metrics['recall']:.4f}\n")
    #                 f.write(f"F1-score: {metrics['f1-score']:.4f}\n")
    #                 f.write(f"Support: {metrics['support']}\n")
    #
    #     print(f"\nResults have been saved to {save_path}")


if __name__ == "__main__":
    losses, f1_macro, f1_weighted = train()

    # Plot the training loss
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss')
    plt.show()




