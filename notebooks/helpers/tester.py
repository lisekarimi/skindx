import math
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
COLOR_MAP = {"red": RED, "orange": YELLOW, "green": GREEN}

class Tester:
    def __init__(self, model, dataloader, label_names, device, title=None):
        self.model = model
        self.dataloader = dataloader
        self.label_names = label_names
        self.device = device
        self.title = title or "Classification Model"
        self.predictions = []
        self.truths = []
        self.confidences = []
        self.correct = []
        self.colors = []
        self.total_samples = 0
        
    def color_for(self, is_correct, confidence):
        """Color coding based on correctness and confidence"""
        if is_correct and confidence > 0.8:
            return "green"  # Correct and confident
        elif is_correct and confidence > 0.5:
            return "orange"  # Correct but low confidence
        else:
            return "red"  # Incorrect
    
    def run_batch(self, images, true_labels):
        """Process a single batch"""
        images = images.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(outputs, dim=1)
            max_confidences = torch.max(probabilities, dim=1)[0]
            
            # Convert to lists
            batch_predictions = [self.label_names[pred.item()] for pred in predicted_classes]
            batch_confidences = max_confidences.cpu().numpy()
            
            # Store results
            for pred, true, conf in zip(batch_predictions, true_labels, batch_confidences):
                is_correct = pred == true
                color = self.color_for(is_correct, conf)
                
                self.predictions.append(pred)
                self.truths.append(true)
                self.confidences.append(conf)
                self.correct.append(is_correct)
                self.colors.append(color)
    
    def chart_confusion_matrix(self, title):
        """Create confusion matrix heatmap"""
        cm = confusion_matrix(self.truths, self.predictions, labels=self.label_names)
        
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_names, yticklabels=self.label_names,
                   cbar_kws={'label': 'Number of Samples'})
        
        plt.title(f'{title}\nConfusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def chart_misclassification_fractions(self, title):
        """Create fractional misclassification bar chart"""
        cm = confusion_matrix(self.truths, self.predictions, labels=self.label_names)
        # Plot fractional incorrect misclassifications
        incorr_fraction = 1 - np.diag(cm) / np.sum(cm, axis=1)
        plt.figure(figsize=(10, 6))
        plt.bar(np.arange(len(self.label_names)), incorr_fraction)
        plt.xlabel('True Label')
        plt.ylabel('Fraction of incorrect predictions')
        plt.title('Fractional Incorrect Misclassifications')
        plt.xticks(np.arange(len(self.label_names)), self.label_names, rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def chart_confidence_distribution(self, title):
        """Create confidence distribution plot"""
        correct_confidences = [conf for conf, correct in zip(self.confidences, self.correct) if correct]
        incorrect_confidences = [conf for conf, correct in zip(self.confidences, self.correct) if not correct]
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(correct_confidences, bins=20, alpha=0.7, color='green', label='Correct Predictions')
        plt.hist(incorrect_confidences, bins=20, alpha=0.7, color='red', label='Incorrect Predictions')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Per-class accuracy
        plt.subplot(1, 2, 2)
        class_accuracies = []
        for label in self.label_names:
            label_correct = [correct for truth, correct in zip(self.truths, self.correct) if truth == label]
            accuracy = sum(label_correct) / len(label_correct) if label_correct else 0
            class_accuracies.append(accuracy)
        
        bars = plt.bar(self.label_names, class_accuracies, color=['green' if acc > 0.8 else 'orange' if acc > 0.6 else 'red' for acc in class_accuracies])
        plt.xlabel('Skin Lesion Type')
        plt.ylabel('Accuracy')
        plt.title('Per-Class Accuracy')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, class_accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def report(self):
        """Generate comprehensive report"""
        accuracy = sum(self.correct) / len(self.correct)
        avg_confidence = sum(self.confidences) / len(self.confidences)
        
        # Count performance categories
        high_conf_correct = sum(1 for correct, conf in zip(self.correct, self.confidences) 
                               if correct and conf > 0.8)
        low_conf_correct = sum(1 for correct, conf in zip(self.correct, self.confidences) 
                              if correct and conf <= 0.8)
        incorrect = sum(1 for correct in self.correct if not correct)
        
        total = len(self.correct)
        
        # Print summary with colors
        title = f"{self.title} - Results Summary"
        print(f"\n{'='*60}")
        print(f"{GREEN}{title}{RESET}")
        print(f"{'='*60}")
        print(f"🎯 Overall Accuracy: {GREEN}{accuracy:.4f} ({accuracy*100:.2f}%){RESET}")
        print(f"📊 Average Confidence: {avg_confidence:.3f}")
        print(f"📈 Total Samples: {total:,}")
        print(f"")
        print(f"Performance Breakdown:")
        print(f"  {GREEN}✅ High Confidence Correct: {high_conf_correct:,} ({high_conf_correct/total*100:.1f}%){RESET}")
        print(f"  {YELLOW}⚠️  Low Confidence Correct:  {low_conf_correct:,} ({low_conf_correct/total*100:.1f}%){RESET}")
        print(f"  {RED}❌ Incorrect Predictions:   {incorrect:,} ({incorrect/total*100:.1f}%){RESET}")
        
        # Create visualizations
        chart_title = f"{self.title} | Accuracy: {accuracy:.3f} | Samples: {total:,}"
        self.chart_confusion_matrix(chart_title)
        self.chart_confidence_distribution(chart_title)
        self.chart_misclassification_fractions(chart_title)
        
        # Print detailed classification report
        print(f"\n{GREEN}Detailed Classification Report:{RESET}")
        print("="*60)
        print(classification_report(self.truths, self.predictions, target_names=self.label_names))
    
    def run(self):
        """Run the complete testing process"""
        self.model.eval()
        print(f"🚀 Starting {self.title} evaluation...")
        
        for batch_idx, (images, true_labels) in enumerate(self.dataloader):
            self.run_batch(images, true_labels)
            self.total_samples += len(true_labels)
            
            if batch_idx % 50 == 0:
                print(f"   Processed {self.total_samples:,} samples...")
        
        print(f"✅ Completed evaluation on {self.total_samples:,} samples")
        self.report()
    
    @classmethod
    def test(cls, model, dataloader, label_names, device, title=None):
        """Convenience method to run complete test"""
        tester = cls(model, dataloader, label_names, device, title)
        tester.run()
        return tester
