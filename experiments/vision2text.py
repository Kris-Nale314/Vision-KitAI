"""
Sample text summarization experiment for Vision-KitAI.

This script demonstrates a complete text summarization experiment
using the Vision-KitAI framework.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from processors.text_processors import TextProcessor
from utils.evaluation import SummaryEvaluator, ExperimentTracker
from utils.data_utils import DataLoader, DataPreprocessor
from experiments.configs import ExperimentConfig, ExperimentRegistry, create_default_config


def run_text_summarization_experiment(config_path=None):
    """
    Run a text summarization experiment.
    
    Args:
        config_path: Path to experiment configuration file
    """
    # Load or create configuration
    if config_path and os.path.exists(config_path):
        config = ExperimentConfig.load(config_path)
    else:
        config = create_default_config(
            name="TextSummarization_Comparison",
            processor_type="text"
        )
        # If no config provided, we'll run a comparison of different methods
        config.description = "Comparison of extractive, abstractive, and boundary-aware summarization"
        
    print(f"Running experiment: {config.name}")
    print(f"Description: {config.description}")
    
    # Initialize components
    text_processor = TextProcessor()
    evaluator = SummaryEvaluator()
    experiment_tracker = ExperimentTracker(config.name, config.output_dir)
    
    # Load dataset
    data_path = config.data_path or os.path.join("data", "text", "news_articles")
    print(f"Loading data from: {data_path}")
    
    # If path is a file, load it directly
    if os.path.isfile(data_path):
        text_documents = [DataLoader.load_text(data_path)]
    else:
        # Otherwise treat as a dataset directory
        dataset = DataLoader.load_dataset(
            data_path,
            split="test",
            processor_type="text",
            sample_size=config.sample_size or 5  # Default to 5 samples if not specified
        )
        text_documents = [sample["text"] for sample in dataset["samples"]]
    
    print(f"Loaded {len(text_documents)} documents")
    
    # Process each document
    for i, document in enumerate(text_documents):
        print(f"\nProcessing document {i+1}/{len(text_documents)}")
        
        document_results = {}
        
        # Preprocess text if needed
        preprocessed_text = DataPreprocessor.preprocess_text(
            document,
            lowercase=False,
            remove_punctuation=False,
            remove_numbers=False
        )
        
        # Apply each summarization method based on config
        if config.method == "extractive" or config.method == "comparison":
            print("Generating extractive summary...")
            extractive_result = text_processor.extractive_summarize(
                preprocessed_text,
                ratio=config.parameters.get("ratio", 0.3),
                min_length=config.parameters.get("min_length", 50),
                max_length=config.parameters.get("max_length", 500)
            )
            document_results["extractive"] = extractive_result
            
            # Evaluate the summary
            print("Evaluating extractive summary...")
            extractive_metrics = evaluator.evaluate_text_summary(
                preprocessed_text, 
                extractive_result["summary"]
            )
            
            # Add to experiment tracker
            experiment_tracker.add_result(
                method_name="Extractive",
                metrics=extractive_metrics,
                parameters={"ratio": config.parameters.get("ratio", 0.3)},
                outputs={"summary": extractive_result["summary"]}
            )
            
        if config.method == "abstractive" or config.method == "comparison":
            print("Generating abstractive summary...")
            abstractive_result = text_processor.abstractive_summarize(
                preprocessed_text,
                max_length=config.parameters.get("max_length", 150),
                min_length=config.parameters.get("min_length", 50),
                do_sample=config.parameters.get("do_sample", False)
            )
            document_results["abstractive"] = abstractive_result
            
            # Evaluate the summary
            print("Evaluating abstractive summary...")
            abstractive_metrics = evaluator.evaluate_text_summary(
                preprocessed_text, 
                abstractive_result["summary"]
            )
            
            # Add to experiment tracker
            experiment_tracker.add_result(
                method_name="Abstractive",
                metrics=abstractive_metrics,
                parameters={"max_length": config.parameters.get("max_length", 150)},
                outputs={"summary": abstractive_result["summary"]}
            )
            
        if config.method == "boundary-aware" or config.method == "comparison":
            print("Generating boundary-aware summary...")
            boundary_result = text_processor.boundary_aware_summarize(
                preprocessed_text,
                method="extractive",
                ratio=config.parameters.get("ratio", 0.3)
            )
            document_results["boundary-aware"] = boundary_result
            
            # Evaluate the summary
            print("Evaluating boundary-aware summary...")
            boundary_metrics = evaluator.evaluate_text_summary(
                preprocessed_text, 
                boundary_result["summary"]
            )
            
            # Add to experiment tracker
            experiment_tracker.add_result(
                method_name="Boundary-Aware",
                metrics=boundary_metrics,
                parameters={"ratio": config.parameters.get("ratio", 0.3)},
                outputs={"summary": boundary_result["summary"]}
            )
            
        if config.method == "entity-focused" or config.method == "comparison":
            print("Generating entity-focused summary...")
            entity_result = text_processor.entity_focused_summarize(
                preprocessed_text,
                method="extractive",
                ratio=config.parameters.get("ratio", 0.3)
            )
            document_results["entity-focused"] = entity_result
            
            # Evaluate the summary
            print("Evaluating entity-focused summary...")
            entity_metrics = evaluator.evaluate_text_summary(
                preprocessed_text, 
                entity_result["summary"]
            )
            
            # Add to experiment tracker
            experiment_tracker.add_result(
                method_name="Entity-Focused",
                metrics=entity_metrics,
                parameters={"ratio": config.parameters.get("ratio", 0.3)},
                outputs={"summary": entity_result["summary"]}
            )
        
        # Save document results
        output_dir = os.path.join(config.output_dir, "summaries")
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, f"document_{i+1}_results.json"), "w") as f:
            json.dump(document_results, f, indent=2)
    
    # Generate comparison visualization
    print("\nGenerating comparison visualization...")
    try:
        import matplotlib.pyplot as plt
        fig = experiment_tracker.visualize_comparison(
            metric_keys=["rouge1", "rouge2", "rougeL", "semantic_similarity.semantic_similarity"]
        )
        
        output_dir = os.path.join(config.output_dir, "visualizations")
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f"{config.name}_comparison.png"))
        plt.close(fig)
    except Exception as e:
        print(f"Error generating visualization: {e}")
    
    # Save experiment results
    print("\nSaving experiment results...")
    results_path = experiment_tracker.save_results()
    print(f"Results saved to: {results_path}")
    
    # Register experiment
    if config_path:
        registry = ExperimentRegistry()
        experiment_id = registry.register_experiment(config, results_path)
        print(f"Experiment registered with ID: {experiment_id}")
    
    print("\nExperiment completed!")
    return results_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a text summarization experiment")
    parser.add_argument(
        "--config", type=str, help="Path to experiment configuration file"
    )
    parser.add_argument(
        "--data", type=str, help="Path to data file or directory"
    )
    parser.add_argument(
        "--method", type=str, choices=["extractive", "abstractive", "boundary-aware", "entity-focused", "comparison"],
        default="comparison", help="Summarization method to use"
    )
    parser.add_argument(
        "--output", type=str, default="output", help="Output directory"
    )
    
    args = parser.parse_args()
    
    # If config file provided, use it
    if args.config:
        run_text_summarization_experiment(args.config)
    else:
        # Otherwise create config from command line args
        config = create_default_config("CLI_Experiment", "text")
        config.method = args.method
        config.data_path = args.data
        config.output_dir = args.output
        
        # Save config to file
        os.makedirs("experiments/configs", exist_ok=True)
        config_path = config.save("experiments/configs")
        
        run_text_summarization_experiment(config_path)