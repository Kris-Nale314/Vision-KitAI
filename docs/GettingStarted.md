# Getting Started with Vision-KitAI

Welcome to Vision-KitAI! This guide will help you set up your local environment and run your first experiments.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/Kris-Nale314/Vision-KitAI.git
cd Vision-KitAI
```

2. **Create a virtual environment**

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

## Project Structure

The repository is organized as follows:

```
vision-kitai/
├── data/                   # Datasets for experimentation
├── processors/             # Core processing algorithms
├── models/                 # Model configurations and weights
├── experiments/            # Structured experiment tracking
├── utils/                  # Shared utilities
├── notebooks/              # Exploratory analysis
├── docs/                   # Documentation
├── demos/                  # Interactive demonstrations
└── output/                 # Generated results
```

## Running Your First Experiment

To run a text summarization experiment:

```bash
python experiments/run_text_summarization.py
```

This will run a comparison of different summarization methods on sample texts and generate reports in the `output` directory.

For more options:

```bash
python experiments/run_text_summarization.py --help
```

## Using the Demo App

To launch the interactive demo:

```bash
streamlit run demos/streamlit_app.py
```

This will start a local web server that you can access in your browser.

## Creating Your Own Experiments

1. Create a new configuration file:

```python
from experiments.configs import create_default_config

config = create_default_config(
    name="MyExperiment",
    processor_type="text"
)
config.save("experiments/configs")
```

2. Modify the parameters as needed:

```python
config.method = "abstractive"
config.parameters = {
    "max_length": 150,
    "min_length": 40
}
```

3. Run the experiment with your configuration:

```bash
python experiments/run_text_summarization.py --config experiments/configs/myexperiment_20230101_120000.yaml
```

## Next Steps

- Explore the notebooks in the `notebooks` directory for more examples
- Add your own data to the `data` directory
- Experiment with different parameters and methods
- Extend the processors with new algorithms

Happy experimenting!