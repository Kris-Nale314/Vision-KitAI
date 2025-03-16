"""
Evaluation utilities for Vision-KitAI.

This module provides metrics and evaluation tools for assessing 
summarization quality across different modalities.
"""

import nltk
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class SummaryEvaluator:
    """Evaluation tools for assessing summarization quality."""
    
    def __init__(self, bert_model_name: str = "microsoft/deberta-base-mnli"):
        """
        Initialize the evaluator.
        
        Args:
            bert_model_name: Transformer model for semantic similarity
        """
        self.bert_model_name = bert_model_name
        self._bert_model = None
        self._bert_tokenizer = None
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def _lazy_load_bert_model(self):
        """Lazy load the BERT model for semantic similarity."""
        if self._bert_model is None:
            self._bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
            self._bert_model = AutoModel.from_pretrained(self.bert_model_name)
    
    def evaluate_text_summary(
        self, 
        original_text: str, 
        summary: str, 
        reference_summary: Optional[str] = None
    ) -> Dict:
        """
        Evaluate a text summary using multiple metrics.
        
        Args:
            original_text: Source text
            summary: Generated summary
            reference_summary: Human-written summary for comparison (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        results = {}
        
        # Content coverage (how much of the original is reflected in the summary)
        results.update(self.calculate_rouge_scores(original_text, summary))
        
        # Conciseness metrics
        original_length = len(original_text.split())
        summary_length = len(summary.split())
        compression_ratio = summary_length / original_length if original_length > 0 else 1.0
        
        results["length"] = {
            "original_words": original_length,
            "summary_words": summary_length,
            "compression_ratio": compression_ratio
        }
        
        # Semantic similarity (how well the summary captures the meaning)
        results["semantic_similarity"] = self.calculate_semantic_similarity(original_text, summary)
        
        # Compare to reference summary if provided
        if reference_summary:
            results["reference_comparison"] = {}
            results["reference_comparison"].update(
                self.calculate_rouge_scores(reference_summary, summary, prefix="ref_")
            )
            results["reference_comparison"]["bert_score"] = self.calculate_bert_score(reference_summary, summary)
        
        return results
    
    def calculate_rouge_scores(
        self, 
        reference: str, 
        summary: str,
        prefix: str = ""
    ) -> Dict:
        """
        Calculate ROUGE scores between reference and summary.
        
        Args:
            reference: Reference text
            summary: Summary to evaluate
            prefix: Prefix for metric names in the output
            
        Returns:
            Dictionary of ROUGE scores
        """
        scores = self.rouge_scorer.score(reference, summary)
        
        return {
            f"{prefix}rouge1": scores["rouge1"].fmeasure,
            f"{prefix}rouge2": scores["rouge2"].fmeasure,
            f"{prefix}rougeL": scores["rougeL"].fmeasure
        }
    
    def calculate_semantic_similarity(
        self, 
        original_text: str, 
        summary: str
    ) -> float:
        """
        Calculate semantic similarity using embeddings.
        
        Args:
            original_text: Original document
            summary: Generated summary
            
        Returns:
            Semantic similarity score (0-1)
        """
        self._lazy_load_bert_model()
        
        # Tokenize and get embeddings for original text and summary
        def get_embedding(text):
            # Split into sentences for better processing
            sentences = nltk.sent_tokenize(text)
            embeddings = []
            
            for sentence in sentences:
                # Skip very short sentences
                if len(sentence.split()) < 3:
                    continue
                    
                inputs = self._bert_tokenizer(sentence, return_tensors="pt", 
                                              padding=True, truncation=True, max_length=512)
                
                with torch.no_grad():
                    outputs = self._bert_model(**inputs)
                
                # Use the [CLS] token embedding as the sentence embedding
                embeddings.append(outputs.last_hidden_state[:, 0, :].numpy())
            
            if not embeddings:
                return np.zeros((1, 768))  # Default embedding size
                
            return np.mean(np.vstack(embeddings), axis=0, keepdims=True)
        
        original_embedding = get_embedding(original_text)
        summary_embedding = get_embedding(summary)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(original_embedding, summary_embedding)[0][0]
        
        return similarity
    
    def calculate_bert_score(
        self, 
        reference: str, 
        candidate: str
    ) -> Dict:
        """
        Calculate BERTScore for evaluation.
        
        Args:
            reference: Reference text
            candidate: Candidate summary
            
        Returns:
            Dictionary with precision, recall, and F1 from BERTScore
        """
        # BERTScore expects lists of references and candidates
        P, R, F1 = bert_score([candidate], [reference], lang="en")
        
        return {
            "precision": P.item(),
            "recall": R.item(),
            "f1": F1.item()
        }
    
    def visualize_metrics(
        self, 
        metrics: List[Dict], 
        names: List[str],
        title: str = "Summary Evaluation Metrics"
    ) -> plt.Figure:
        """
        Visualize evaluation metrics for multiple summarization methods.
        
        Args:
            metrics: List of metric dictionaries from evaluate_text_summary
            names: Names of the summarization methods
            title: Plot title
            
        Returns:
            Matplotlib figure with visualization
        """
        if not metrics or not names:
            raise ValueError("Empty metrics or names list")
            
        if len(metrics) != len(names):
            raise ValueError("Length of metrics and names must match")
        
        # Extract common metrics
        metric_keys = ["rouge1", "rouge2", "rougeL", "semantic_similarity"]
        metric_names = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "Semantic Similarity"]
        
        # Create dataframe for plotting
        data = []
        for name, metric in zip(names, metrics):
            row = [name]
            for key in metric_keys:
                row.append(metric.get(key, metric.get("reference_comparison", {}).get(key, 0)))
            data.append(row)
            
        df = pd.DataFrame(data, columns=["Method"] + metric_names)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        df.set_index("Method").plot(kind="bar", ax=ax)
        
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.legend(loc="lower right")
        
        plt.tight_layout()
        return fig


class ExperimentTracker:
    """Track and compare summarization experiments."""
    
    def __init__(self, experiment_name: str, output_dir: str = "output/metrics"):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the experiment
            output_dir: Directory to save results
        """
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.results = []
        
    def add_result(
        self, 
        method_name: str, 
        metrics: Dict, 
        parameters: Dict = None,
        outputs: Dict = None
    ):
        """
        Add an experiment result.
        
        Args:
            method_name: Name of the summarization method
            metrics: Evaluation metrics
            parameters: Parameters used for the method
            outputs: Generated outputs (summaries, etc.)
        """
        self.results.append({
            "method": method_name,
            "metrics": metrics,
            "parameters": parameters or {},
            "outputs": outputs or {}
        })
    
    def compare_methods(self, metric_keys: List[str] = None) -> pd.DataFrame:
        """
        Compare different methods based on selected metrics.
        
        Args:
            metric_keys: List of metric keys to compare (default: all metrics)
            
        Returns:
            DataFrame with method comparison
        """
        if not self.results:
            return pd.DataFrame()
            
        # Determine metrics to include
        if metric_keys is None:
            # Get all metrics from the first result
            all_metrics = list(self.results[0]["metrics"].keys())
            # Filter out nested metrics
            metric_keys = [k for k in all_metrics if not isinstance(self.results[0]["metrics"][k], dict)]
        
        # Create comparison dataframe
        data = []
        methods = []
        
        for result in self.results:
            methods.append(result["method"])
            row = []
            
            for key in metric_keys:
                # Handle nested metrics
                if "." in key:
                    parts = key.split(".")
                    value = result["metrics"]
                    for part in parts:
                        if isinstance(value, dict) and part in value:
                            value = value[part]
                        else:
                            value = None
                            break
                else:
                    value = result["metrics"].get(key)
                
                row.append(value)
            
            data.append(row)
        
        return pd.DataFrame(data, index=methods, columns=metric_keys)
    
    def visualize_comparison(
        self, 
        metric_keys: List[str] = None,
        title: str = None
    ) -> plt.Figure:
        """
        Create visualization comparing methods.
        
        Args:
            metric_keys: List of metric keys to visualize
            title: Plot title
            
        Returns:
            Matplotlib figure with visualization
        """
        df = self.compare_methods(metric_keys)
        
        if df.empty:
            raise ValueError("No results to visualize")
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        df.plot(kind="bar", ax=ax)
        
        if title is None:
            title = f"Comparison of Methods - {self.experiment_name}"
            
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.legend(loc="lower right")
        
        plt.tight_layout()
        return fig
    
    def save_results(self, filename: str = None):
        """
        Save experiment results to file.
        
        Args:
            filename: Output filename (default: based on experiment name)
        """
        import os
        import json
        from datetime import datetime
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.experiment_name}_{timestamp}.json"
        
        output_path = os.path.join(self.output_dir, filename)
        
        # Save as JSON
        with open(output_path, "w") as f:
            json.dump({
                "experiment_name": self.experiment_name,
                "timestamp": datetime.now().isoformat(),
                "results": self.results
            }, f, indent=2)
            
        return output_path