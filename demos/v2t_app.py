"""
Streamlit demo app for Vision-KitAI.

This app provides an interactive interface for exploring the text summarization
capabilities of Vision-KitAI.
"""

import os
import sys
import json
import time
import streamlit as st
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from processors.text_processors import TextProcessor
from utils.evaluation import SummaryEvaluator
from utils.data_utils import DataLoader, DataPreprocessor

# Set page config
st.set_page_config(
    page_title="Vision-KitAI: Text Summarization",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def load_processor():
    return TextProcessor()

@st.cache_resource
def load_evaluator():
    return SummaryEvaluator()

text_processor = load_processor()
evaluator = load_evaluator()

# Header
st.title("Vision-KitAI: Text Summarization")
st.markdown("""
This demo showcases the text summarization capabilities of Vision-KitAI.
You can input your own text or choose a sample, then compare different summarization methods.
""")

# Sidebar
st.sidebar.header("Vision-KitAI")
st.sidebar.image("docs/images/logo.png", width=150)
st.sidebar.markdown("### Summarization Settings")

# Input options
input_option = st.sidebar.radio(
    "Input Source",
    ["Sample Text", "Custom Text", "Upload File"]
)

# Sample texts
sample_texts = {
    "News Article": """The European Space Agency (ESA) has successfully launched the Euclid space telescope, designed to explore the dark universe. The telescope, named after the ancient Greek mathematician, will map the geometry of the Universe and help astronomers better understand dark matter and dark energy, which together make up 95% of the cosmos but remain largely mysterious.

The Euclid telescope was launched from Cape Canaveral in Florida on a SpaceX Falcon 9 rocket. It will orbit the Sun at a distance of about 1.5 million kilometers from Earth. From there, it will observe billions of galaxies across more than a third of the sky to create what has been called a "cosmic movie" of how the Universe has evolved over the past 10 billion years.

"We're trying to understand what makes the Universe tick, and a lot of that is about understanding dark energy and dark matter," said Dr. Jason Rhodes, a project scientist for Euclid at NASA's Jet Propulsion Laboratory.

Dark matter is invisible but has been detected through its gravitational effects on visible matter. Dark energy is even more mysterious and is believed to be responsible for the accelerating expansion of the Universe.

The Euclid mission is expected to last at least six years and will produce about 26 petabytes of data - equivalent to over 5 million DVDs of information.""",
    
    "Scientific Paper Abstract": """Transformer models have recently achieved state-of-the-art performance on a variety of natural language processing tasks. However, these models typically require large amounts of compute for both training and inference, limiting their broader applicability. In this work, we present a new approach to reduce the computational requirements of Transformers. Our method, which we call Attention Sinks, leverages the insight that attention patterns in Transformers often exhibit a concentration of attention weights on a small subset of tokens. By identifying and preserving these "attention sinks" while pruning less important connections, we can significantly reduce both memory usage and computational requirements without sacrificing model performance. We evaluate our approach on several benchmark tasks including language modeling, machine translation, and text summarization. Our results show that Attention Sinks can reduce inference time by up to 40% while maintaining within 1% of the original model's performance. This approach provides a practical way to deploy powerful Transformer models in resource-constrained environments.""",
    
    "Financial Report": """Apple Inc. (NASDAQ: AAPL) today announced financial results for its fiscal 2023 third quarter ended July 1, 2023. The Company posted quarterly revenue of $81.8 billion, down 1 percent year over year, and quarterly earnings per diluted share of $1.26, up 5 percent year over year.

"We are happy to report that we had an all-time revenue record in Services during the June quarter, driven by over 1 billion paid subscriptions, and we saw continued strength in emerging markets thanks to robust sales of iPhone," said Tim Cook, Apple's CEO. "From education to the environment, we are continuing to advance our values, while championing innovation that enriches the lives of our customers and leaves the world better than we found it."

"Our June quarter year-over-year business performance improved from the March quarter, and we generated strong operating cash flow of $26 billion while returning over $24 billion to shareholders during the quarter," said Luca Maestri, Apple's CFO. "Given our confidence in Apple's future and the value we see in our stock, our Board has authorized an additional $90 billion for share repurchases. We are also raising our quarterly dividend for the eleventh year in a row."

Apple's board of directors has declared a cash dividend of $0.24 per share of the Company's common stock. The dividend is payable on August 17, 2023 to shareholders of record as of the close of business on August 14, 2023."""
}

# Method selection
summarization_methods = st.sidebar.multiselect(
    "Summarization Methods",
    ["Extractive", "Abstractive", "Boundary-Aware", "Entity-Focused"],
    default=["Extractive", "Abstractive"]
)

# Extractive settings
if "Extractive" in summarization_methods:
    st.sidebar.markdown("### Extractive Settings")
    extractive_ratio = st.sidebar.slider(
        "Compression Ratio",
        min_value=0.1,
        max_value=0.5,
        value=0.3,
        step=0.05,
        help="Proportion of the original text to keep (lower means more compression)"
    )

# Abstractive settings
if "Abstractive" in summarization_methods:
    st.sidebar.markdown("### Abstractive Settings")
    max_length = st.sidebar.slider(
        "Maximum Length (tokens)",
        min_value=50,
        max_value=250,
        value=150,
        step=10,
        help="Maximum length of the generated summary"
    )
    min_length = st.sidebar.slider(
        "Minimum Length (tokens)",
        min_value=10,
        max_value=100,
        value=50,
        step=10,
        help="Minimum length of the generated summary"
    )
    
# Get input text
input_text = ""

if input_option == "Sample Text":
    sample_choice = st.selectbox("Select a sample text", list(sample_texts.keys()))
    input_text = sample_texts[sample_choice]
    
elif input_option == "Custom Text":
    input_text = st.text_area("Enter text to summarize", height=300)
    
elif input_option == "Upload File":
    uploaded_file = st.file_uploader("Upload a text file", type=["txt", "md", "csv", "json"])
    if uploaded_file is not None:
        input_text = uploaded_file.getvalue().decode("utf-8")

# Display input text
if input_text:
    with st.expander("Input Text", expanded=False):
        st.markdown(input_text)
    
    # Show text stats
    word_count = len(input_text.split())
    sentence_count = len(input_text.split("."))
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Word Count", word_count)
    col2.metric("Sentence Count", sentence_count)
    col3.metric("Avg Words per Sentence", round(word_count / max(1, sentence_count), 1))
    
    # Generate summaries button
    if st.button("Generate Summaries"):
        with st.spinner("Generating summaries..."):
            # Container for results
            results = {}
            
            # Generate each selected summary
            if "Extractive" in summarization_methods:
                with st.status("Generating extractive summary...") as status:
                    start_time = time.time()
                    extractive_result = text_processor.extractive_summarize(
                        input_text,
                        ratio=extractive_ratio
                    )
                    end_time = time.time()
                    status.update(label="Extractive summary completed!", state="complete")
                    
                    results["Extractive"] = {
                        "summary": extractive_result["summary"],
                        "time": end_time - start_time,
                        "metrics": {
                            "compression_ratio": extractive_result["compression_ratio"],
                            "sentence_count": extractive_result["sentence_count"]
                        }
                    }
            
            if "Abstractive" in summarization_methods:
                with st.status("Generating abstractive summary...") as status:
                    start_time = time.time()
                    abstractive_result = text_processor.abstractive_summarize(
                        input_text,
                        max_length=max_length,
                        min_length=min_length
                    )
                    end_time = time.time()
                    status.update(label="Abstractive summary completed!", state="complete")
                    
                    results["Abstractive"] = {
                        "summary": abstractive_result["summary"],
                        "time": end_time - start_time,
                        "metrics": {
                            "compression_ratio": abstractive_result["compression_ratio"],
                            "model": abstractive_result["model"]
                        }
                    }
            
            if "Boundary-Aware" in summarization_methods:
                with st.status("Generating boundary-aware summary...") as status:
                    start_time = time.time()
                    boundary_result = text_processor.boundary_aware_summarize(
                        input_text,
                        method="extractive",
                        ratio=extractive_ratio
                    )
                    end_time = time.time()
                    status.update(label="Boundary-aware summary completed!", state="complete")
                    
                    results["Boundary-Aware"] = {
                        "summary": boundary_result["summary"],
                        "time": end_time - start_time,
                        "metrics": {
                            "section_count": boundary_result["section_count"]
                        }
                    }
            
            if "Entity-Focused" in summarization_methods:
                with st.status("Generating entity-focused summary...") as status:
                    start_time = time.time()
                    entity_result = text_processor.entity_focused_summarize(
                        input_text,
                        method="extractive",
                        ratio=extractive_ratio
                    )
                    end_time = time.time()
                    status.update(label="Entity-focused summary completed!", state="complete")
                    
                    results["Entity-Focused"] = {
                        "summary": entity_result["summary"],
                        "time": end_time - start_time,
                        "metrics": {
                            "entities": ", ".join(entity_result["entities"][:5]) + 
                                       ("..." if len(entity_result["entities"]) > 5 else "")
                        }
                    }
            
            # Display results
            st.markdown("## Summary Results")
            
            # Create tabs for each method
            tabs = st.tabs(list(results.keys()))
            
            for i, (method, data) in enumerate(results.items()):
                with tabs[i]:
                    st.markdown(f"### {method} Summary")
                    st.markdown(data["summary"])
                    
                    # Display metrics
                    st.markdown("#### Metrics")
                    metrics_cols = st.columns(3)
                    
                    metrics_cols[0].metric("Processing Time", f"{data['time']:.2f}s")
                    metrics_cols[1].metric("Word Count", len(data["summary"].split()))
                    
                    for j, (metric_name, metric_value) in enumerate(data["metrics"].items()):
                        metrics_cols[j % 3].metric(
                            metric_name.replace("_", " ").title(),
                            metric_value if not isinstance(metric_value, float) else f"{metric_value:.2f}"
                        )
                    
                    # Calculate semantic similarity
                    with st.spinner("Calculating semantic similarity..."):
                        similarity = evaluator.calculate_semantic_similarity(input_text, data["summary"])
                        metrics_cols[2].metric("Semantic Similarity", f"{similarity:.2f}")
            
            # Comparison view
            st.markdown("## Comparison")
            
            # Word count comparison
            word_counts = {method: len(data["summary"].split()) for method, data in results.items()}
            word_count_ratio = {method: count / word_count for method, count in word_counts.items()}
            
            st.bar_chart(word_count_ratio)
            
            # Side-by-side comparison
            if len(results) > 1:
                st.markdown("### Side-by-Side Comparison")
                cols = st.columns(len(results))
                
                for i, (method, data) in enumerate(results.items()):
                    with cols[i]:
                        st.markdown(f"**{method}**")
                        st.markdown(data["summary"])
            
            # Save results
            results_dir = os.path.join("output", "demo_results")
            os.makedirs(results_dir, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(results_dir, f"summary_comparison_{timestamp}.json")
            
            with open(results_file, "w") as f:
                json.dump({
                    "input_text": input_text,
                    "word_count": word_count,
                    "results": {
                        method: {
                            "summary": data["summary"],
                            "time": data["time"],
                            "metrics": data["metrics"]
                        } for method, data in results.items()
                    }
                }, f, indent=2)
                
            st.success(f"Results saved to {results_file}")
else:
    st.info("Please enter or select text to summarize")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center">
        <p>Vision-KitAI: A playground for exploring vision AI technologies</p>
        <p>Created by Kris | <a href="https://github.com/Kris-Nale314/Vision-KitAI">GitHub</a></p>
    </div>
    """,
    unsafe_allow_html=True
)