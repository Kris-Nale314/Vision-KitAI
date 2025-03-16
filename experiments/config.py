"""
Experiment configuration utilities for Vision-KitAI.

This module provides tools for defining, loading, and tracking experiment
configurations to ensure reproducibility and systematic comparison.
"""

import os
import json
import yaml
import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    
    # Basic information
    name: str
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    author: str = "Kris"
    
    # Experiment parameters
    processor_type: str = "text"  # text, image, video, multimodal
    method: str = "extractive"    # specific method within the processor type
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Data settings
    dataset: str = ""
    data_path: str = ""
    sample_size: int = 0  # 0 means use all available data
    
    # Evaluation settings
    metrics: List[str] = field(default_factory=list)
    baseline_method: Optional[str] = None
    
    # Output settings
    output_dir: str = "output"
    save_artifacts: bool = True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, path: str) -> str:
        """
        Save configuration to file.
        
        Args:
            path: Directory to save the config file
            
        Returns:
            Path to the saved config file
        """
        os.makedirs(path, exist_ok=True)
        
        # Generate filename from experiment name
        safe_name = self.name.replace(" ", "_").lower()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_name}_{timestamp}.yaml"
        
        full_path = os.path.join(path, filename)
        
        # Save as YAML
        with open(full_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
            
        return full_path
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'ExperimentConfig':
        """
        Create config from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            ExperimentConfig object
        """
        return cls(**config_dict)
    
    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """
        Load configuration from file.
        
        Args:
            path: Path to config file (YAML or JSON)
            
        Returns:
            ExperimentConfig object
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
            
        # Determine file type and load accordingly
        if path.endswith(".yaml") or path.endswith(".yml"):
            with open(path, "r") as f:
                config_dict = yaml.safe_load(f)
        elif path.endswith(".json"):
            with open(path, "r") as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path}")
            
        return cls.from_dict(config_dict)


class ExperimentRegistry:
    """Registry for tracking and comparing experiments."""
    
    def __init__(self, registry_dir: str = "experiments/registry"):
        """
        Initialize experiment registry.
        
        Args:
            registry_dir: Directory for storing experiment registry
        """
        self.registry_dir = registry_dir
        self.registry_file = os.path.join(registry_dir, "experiment_registry.json")
        self.experiments = self._load_registry()
        
    def _load_registry(self) -> Dict:
        """Load experiment registry from file."""
        os.makedirs(self.registry_dir, exist_ok=True)
        
        if os.path.exists(self.registry_file):
            with open(self.registry_file, "r") as f:
                return json.load(f)
        else:
            return {"experiments": []}
            
    def _save_registry(self):
        """Save experiment registry to file."""
        with open(self.registry_file, "w") as f:
            json.dump(self.experiments, f, indent=2)
            
    def register_experiment(
        self, 
        config: ExperimentConfig, 
        results_path: Optional[str] = None
    ) -> str:
        """
        Register an experiment in the registry.
        
        Args:
            config: Experiment configuration
            results_path: Path to experiment results (optional)
            
        Returns:
            Experiment ID
        """
        # Generate ID for experiment
        experiment_id = f"{config.name.replace(' ', '_').lower()}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create experiment entry
        experiment_entry = {
            "id": experiment_id,
            "name": config.name,
            "description": config.description,
            "created_at": config.created_at,
            "author": config.author,
            "processor_type": config.processor_type,
            "method": config.method,
            "dataset": config.dataset,
            "config_path": None,  # Will be set after saving config
            "results_path": results_path
        }
        
        # Save config file
        config_path = config.save(os.path.join(self.registry_dir, "configs"))
        experiment_entry["config_path"] = config_path
        
        # Add to registry
        self.experiments["experiments"].append(experiment_entry)
        self._save_registry()
        
        return experiment_id
        
    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """
        Get experiment from registry.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Experiment entry or None if not found
        """
        for experiment in self.experiments["experiments"]:
            if experiment["id"] == experiment_id:
                return experiment
                
        return None
        
    def update_results(self, experiment_id: str, results_path: str):
        """
        Update results path for an experiment.
        
        Args:
            experiment_id: Experiment ID
            results_path: Path to experiment results
        """
        for experiment in self.experiments["experiments"]:
            if experiment["id"] == experiment_id:
                experiment["results_path"] = results_path
                self._save_registry()
                return
                
        raise ValueError(f"Experiment not found: {experiment_id}")
        
    def list_experiments(
        self, 
        processor_type: Optional[str] = None,
        method: Optional[str] = None,
        dataset: Optional[str] = None
    ) -> List[Dict]:
        """
        List experiments with optional filtering.
        
        Args:
            processor_type: Filter by processor type
            method: Filter by method
            dataset: Filter by dataset
            
        Returns:
            List of matching experiment entries
        """
        results = []
        
        for experiment in self.experiments["experiments"]:
            # Apply filters
            if processor_type and experiment["processor_type"] != processor_type:
                continue
                
            if method and experiment["method"] != method:
                continue
                
            if dataset and experiment["dataset"] != dataset:
                continue
                
            results.append(experiment)
            
        return results
        
    def compare_experiments(self, experiment_ids: List[str]) -> Dict:
        """
        Load and compare results from multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            
        Returns:
            Dictionary with comparison data
        """
        comparison = {
            "experiments": [],
            "metrics": {}
        }
        
        for exp_id in experiment_ids:
            experiment = self.get_experiment(exp_id)
            
            if not experiment:
                print(f"Warning: Experiment {exp_id} not found")
                continue
                
            if not experiment["results_path"] or not os.path.exists(experiment["results_path"]):
                print(f"Warning: Results not found for experiment {exp_id}")
                continue
                
            # Load results
            with open(experiment["results_path"], "r") as f:
                results = json.load(f)
                
            # Add to comparison
            comparison["experiments"].append({
                "id": experiment["id"],
                "name": experiment["name"],
                "method": experiment["method"],
                "results": results
            })
            
            # Collect metrics
            for metric_name, metric_value in results.get("metrics", {}).items():
                if metric_name not in comparison["metrics"]:
                    comparison["metrics"][metric_name] = []
                    
                comparison["metrics"][metric_name].append({
                    "experiment_id": experiment["id"],
                    "experiment_name": experiment["name"],
                    "value": metric_value
                })
                
        return comparison


def create_default_config(name: str, processor_type: str = "text") -> ExperimentConfig:
    """
    Create default configuration for different processor types.
    
    Args:
        name: Experiment name
        processor_type: Type of processor (text, image, video, multimodal)
        
    Returns:
        Default ExperimentConfig
    """
    config = ExperimentConfig(
        name=name,
        processor_type=processor_type,
        description=f"Default {processor_type} processing experiment"
    )
    
    # Set default settings based on processor type
    if processor_type == "text":
        config.method = "extractive"
        config.parameters = {
            "ratio": 0.3,
            "min_length": 50,
            "max_length": 500
        }
        config.metrics = ["rouge1", "rouge2", "rougeL", "semantic_similarity"]
        
    elif processor_type == "image":
        config.method = "captioning"
        config.parameters = {
            "max_tokens": 50,
            "num_beams": 4
        }
        config.metrics = ["bleu", "meteor", "cider"]
        
    elif processor_type == "video":
        config.method = "keyframe"
        config.parameters = {
            "threshold": 0.5,
            "max_frames": 10
        }
        config.metrics = ["coverage", "diversity"]
        
    elif processor_type == "multimodal":
        config.method = "text_image_fusion"
        config.parameters = {
            "text_weight": 0.6,
            "image_weight": 0.4
        }
        config.metrics = ["rouge1", "rouge2", "semantic_similarity", "visual_coverage"]
        
    return config