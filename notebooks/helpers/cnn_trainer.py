import os, tempfile
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import mlflow
import mlflow.pytorch

def train_cnn_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    data_balancing,
    scheduler=None,
    num_epochs=50,
    patience=10,
    image_size=32,
    data_augmentation=False,
    notes="",
    user_name="unknown",
    device=None,
    label_names=None
):
    # --- Safety checks ---
    if label_names is None:
        raise ValueError("label_names must be provided (list of class names).")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Auto-generate run name with model info
    run_name = f"{model.__class__.__name__}"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("user", user_name)
        mlflow.set_tag("notes", notes)

        # Log hyperparameters
        mlflow.log_params({
            "model_type": model.__class__.__name__,
            "epochs": num_epochs,
            "batch_size": train_loader.batch_size,
            "image_size": image_size,
            "data_augmentation": data_augmentation,
            "patience": patience,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "optimizer": optimizer.__class__.__name__,
            "scheduler": scheduler.__class__.__name__ if scheduler else "None",
            "loss_function": criterion.__class__.__name__,
            "data_balancing": data_balancing,
            "transfer_learning": hasattr(model, 'fc'),
            "frozen_layers": "None",
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "test_samples": len(test_loader.dataset)
        })

        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        best_epoch = 0

        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            for images, labels in progress_bar:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                current_loss = train_loss / (progress_bar.n + 1)
                current_acc = (train_correct / train_total) * 100
                progress_bar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'accuracy': f'{current_acc:.2f}%'
                })

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            # Calculate final metrics
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            train_loss_avg = train_loss / len(train_loader)
            val_loss_avg = val_loss / len(val_loader)
            # Add scheduler step here
            if scheduler:
                scheduler.step(val_loss_avg)

            # Store metrics
            train_losses.append(train_loss_avg)
            val_losses.append(val_loss_avg)
            train_accs.append(train_acc * 100)
            val_accs.append(val_acc * 100)

            # Log metrics to MLflow
            mlflow.log_metrics({
                "train_loss": train_loss_avg,
                "val_loss": val_loss_avg,
                "train_accuracy": train_acc * 100,
                "val_accuracy": val_acc * 100,
                "learning_rate": optimizer.param_groups[0]['lr']
            }, step=epoch)

            # Print metrics
            print(f'Epoch {epoch+1}: loss: {train_loss_avg:.4f} - accuracy: {train_acc*100:.2f}% - val_loss: {val_loss_avg:.4f} - val_accuracy: {val_acc*100:.2f}%')

            # Early stopping check
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                best_epoch = epoch + 1
                print('Validation loss improved. Best model updated.')
            else:
                patience_counter += 1
                print(f'No improvement. Patience: {patience_counter}/{patience}')

                if patience_counter >= patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f'Best model from epoch {best_epoch} loaded')

        # Test evaluation
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_acc = (test_correct / test_total) * 100
        test_loss_avg = test_loss / len(test_loader)

        print(f"\nTest Results:")
        print(f"Test Loss: {test_loss_avg:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")

        # Generate and log confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_names, yticklabels=label_names)
        plt.title('Confusion Matrix - Simple CNN')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), "confusion_matrix.png")
        plt.show()

        # Plot fractional incorrect misclassifications
        incorr_fraction = 1 - np.diag(cm) / np.sum(cm, axis=1)
        plt.figure(figsize=(10, 6))
        n_classes = len(label_names)
        plt.bar(np.arange(n_classes), incorr_fraction)
        plt.xlabel('True Label')
        plt.ylabel('Fraction of incorrect predictions')
        plt.title('Fractional Incorrect Misclassifications')
        plt.xticks(np.arange(n_classes), label_names, rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), "misclassification_fractions.png")
        plt.show()

        # Generate and log classification report
        report = classification_report(all_labels, all_predictions, target_names=label_names, output_dict=True, zero_division=0)
        report_text = classification_report(all_labels, all_predictions, target_names=label_names, zero_division=0)

        # Print classification report
        print("\nClassification Report:")
        print("="*50)
        print(report_text)
        print("="*50)

        # Save classification report as artifact
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(report_text)
            temp_file = f.name
        mlflow.log_artifact(temp_file, "classification_report.txt")

        # Log per-class metrics
        for class_name in label_names:
            if class_name in report:
                mlflow.log_metrics({
                    f"{class_name}_precision": report[class_name]['precision'],
                    f"{class_name}_recall": report[class_name]['recall'],
                    f"{class_name}_f1_score": report[class_name]['f1-score']
                })

        # Create and log training history plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        epochs_range = range(1, len(train_losses) + 1)

        ax1.plot(epochs_range, train_losses, 'y-', label='Training Loss', linewidth=2)
        ax1.plot(epochs_range, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs_range, train_accs, 'y-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs_range, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        mlflow.log_figure(fig, "training_history.png")
        plt.show()

        # Log final metrics
        mlflow.log_metrics({
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "total_epochs": len(train_losses),
            "best_train_accuracy": max(train_accs),
            "best_val_accuracy": max(val_accs),
            "test_loss": test_loss_avg,
            "test_accuracy": test_acc
        })

        # Log model to MLflow
        mlflow.pytorch.log_model(model, artifact_path=model.__class__.__name__)

        # Training completion message
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY! 🎉")
        print("="*60)
        print(f"✅ Total epochs trained: {len(train_losses)}")
        print(f"✅ Best epoch: {best_epoch}")
        print(f"✅ Final test accuracy: {test_acc:.2f}%")
        print(f"✅ Model saved to MLflow")
        print(f"✅ All artifacts logged successfully")
        print("="*60)
        print("Training and evaluation completed!")

    return train_losses, val_losses, train_accs, val_accs, test_loss_avg, test_acc