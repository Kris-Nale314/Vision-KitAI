"""
Text processing module for Vision-KitAI.

This module provides text summarization capabilities as the foundation
for our staged approach to multimodal summarization.
"""

import re
import nltk
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Download NLTK resources if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


class TextProcessor:
    """
    Provides text summarization capabilities using various techniques.
    
    This processor implements both extractive and abstractive summarization
    methods as a foundation for multimodal summarization.
    """
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """
        Initialize the text processor.
        
        Args:
            model_name: Hugging Face model identifier for abstractive summarization
        """
        self.model_name = model_name
        self._abstractive_summarizer = None
        self._tokenizer = None
        
    def _lazy_load_model(self):
        """Lazy load the abstractive summarization model when needed."""
        if self._abstractive_summarizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self._abstractive_summarizer = pipeline(
                "summarization", 
                model=model, 
                tokenizer=self._tokenizer
            )
    
    def extractive_summarize(
        self, 
        text: str, 
        ratio: float = 0.3, 
        min_length: int = 40, 
        max_length: int = 600
    ) -> Dict:
        """
        Create an extractive summary by selecting the most important sentences.
        
        Args:
            text: Input text to summarize
            ratio: Proportion of the original text to keep (0.0-1.0)
            min_length: Minimum summary length in characters
            max_length: Maximum summary length in characters
            
        Returns:
            Dictionary containing the summary and metadata
        """
        # Preprocess text
        sentences = nltk.sent_tokenize(text)
        
        # Handle very short texts
        if len(sentences) <= 2:
            return {
                "summary": text,
                "method": "extractive",
                "sentence_count": len(sentences),
                "compression_ratio": 1.0,
                "selected_indices": list(range(len(sentences)))
            }
            
        # Calculate sentence importance using TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')
        sentence_vectors = vectorizer.fit_transform(sentences)
        
        # Calculate sentence scores
        sentence_scores = np.sum(sentence_vectors.toarray(), axis=1)
        
        # Determine how many sentences to keep
        num_sentences = max(2, min(
            int(len(sentences) * ratio),
            int(max_length / (sum(len(s) for s in sentences) / len(sentences)))
        ))
        
        # Get indices of top sentences
        top_indices = np.argsort(sentence_scores)[-num_sentences:]
        # Sort indices to maintain original order
        top_indices = sorted(top_indices)
        
        # Create summary by joining selected sentences
        selected_sentences = [sentences[i] for i in top_indices]
        summary = " ".join(selected_sentences)
        
        # Calculate compression ratio
        original_length = len(text)
        summary_length = len(summary)
        compression = summary_length / original_length if original_length > 0 else 1.0
        
        return {
            "summary": summary,
            "method": "extractive",
            "sentence_count": len(selected_sentences),
            "compression_ratio": compression,
            "selected_indices": top_indices.tolist() if isinstance(top_indices, np.ndarray) else top_indices
        }
        
    def abstractive_summarize(
        self, 
        text: str, 
        max_length: int = 150, 
        min_length: int = 40, 
        do_sample: bool = False
    ) -> Dict:
        """
        Create an abstractive summary using a transformer model.
        
        Args:
            text: Input text to summarize
            max_length: Maximum summary length in tokens
            min_length: Minimum summary length in tokens
            do_sample: Whether to use sampling for generation
            
        Returns:
            Dictionary containing the summary and metadata
        """
        # Lazy load the model when first needed
        self._lazy_load_model()
        
        # Truncate text if needed (for model max context)
        input_ids = self._tokenizer.encode(text, return_tensors="pt")
        max_model_length = self._tokenizer.model_max_length
        
        if input_ids.shape[1] > max_model_length:
            # Truncate while trying to maintain sentence boundaries
            sentences = nltk.sent_tokenize(text)
            truncated_text = ""
            for sentence in sentences:
                if len(self._tokenizer.encode(truncated_text + sentence)) <= max_model_length:
                    truncated_text += sentence + " "
                else:
                    break
            text = truncated_text
        
        # Generate summary
        summary_result = self._abstractive_summarizer(
            text, 
            max_length=max_length, 
            min_length=min_length,
            do_sample=do_sample,
            return_text=True
        )
        
        # Calculate compression ratio
        original_length = len(text)
        summary_length = len(summary_result[0]["summary_text"])
        compression = summary_length / original_length if original_length > 0 else 1.0
        
        return {
            "summary": summary_result[0]["summary_text"],
            "method": "abstractive",
            "model": self.model_name,
            "compression_ratio": compression
        }
    
    def boundary_aware_summarize(
        self, 
        text: str, 
        sections: List[Dict] = None, 
        method: str = "extractive",
        **kwargs
    ) -> Dict:
        """
        Create a summary that respects document boundaries.
        
        Args:
            text: Input text to summarize
            sections: List of document sections with their boundaries
            method: Summarization method ('extractive' or 'abstractive')
            **kwargs: Additional parameters for the underlying summarization method
            
        Returns:
            Dictionary containing the summary and metadata
        """
        # If no sections provided, try to detect them
        if sections is None:
            sections = self._detect_sections(text)
        
        # Summarize each section
        section_summaries = []
        for section in sections:
            section_text = section.get("text", "")
            heading = section.get("heading", "")
            
            # Skip very short sections
            if len(section_text.split()) < 10:
                section_summaries.append({"heading": heading, "summary": section_text})
                continue
                
            # Calculate appropriate length for this section
            relative_size = len(section_text) / len(text) if len(text) > 0 else 0
            
            if method == "extractive":
                # Adjust ratio based on section importance (headings usually need less compression)
                is_heading_section = heading.strip() != "" and len(section_text.split()) < 100
                ratio = 0.8 if is_heading_section else 0.3
                
                summary_result = self.extractive_summarize(
                    section_text, 
                    ratio=ratio,
                    **kwargs
                )
                summary = summary_result["summary"]
                
            else:  # abstractive
                # Scale max/min length to section size
                max_length = kwargs.get("max_length", 150)
                min_length = kwargs.get("min_length", 40)
                
                scaled_max = max(min_length, int(max_length * relative_size))
                scaled_min = min(scaled_max, min_length)
                
                summary_result = self.abstractive_summarize(
                    section_text,
                    max_length=scaled_max,
                    min_length=scaled_min,
                    **kwargs
                )
                summary = summary_result["summary"]
            
            section_summaries.append({"heading": heading, "summary": summary})
        
        # Combine section summaries
        combined_summary = ""
        for section_summary in section_summaries:
            heading = section_summary["heading"]
            summary = section_summary["summary"]
            
            if heading:
                combined_summary += f"{heading}\n\n"
            combined_summary += f"{summary}\n\n"
        
        return {
            "summary": combined_summary.strip(),
            "method": f"boundary_aware_{method}",
            "section_count": len(sections),
            "section_summaries": section_summaries
        }
        
    def _detect_sections(self, text: str) -> List[Dict]:
        """
        Detect document sections based on formatting cues.
        
        Args:
            text: Document text
            
        Returns:
            List of detected sections with their text and headings
        """
        # Look for common heading patterns
        heading_patterns = [
            r'^#{1,6}\s+(.+)$',  # Markdown headings
            r'^(\d+\..*?)$',  # Numbered sections
            r'^([A-Z][A-Za-z\s]+:)$',  # Capitalized phrases with colon
            r'^([A-Z][A-Z\s]+)$'  # ALL CAPS lines
        ]
        
        lines = text.split('\n')
        sections = []
        current_heading = ""
        current_content = []
        
        for line in lines:
            is_heading = False
            
            # Check if line matches any heading pattern
            for pattern in heading_patterns:
                match = re.match(pattern, line.strip(), re.MULTILINE)
                if match:
                    # If we have content from a previous section, save it
                    if current_content:
                        sections.append({
                            "heading": current_heading,
                            "text": '\n'.join(current_content).strip()
                        })
                        current_content = []
                    
                    current_heading = line.strip()
                    is_heading = True
                    break
            
            if not is_heading:
                current_content.append(line)
        
        # Add the last section
        if current_content:
            sections.append({
                "heading": current_heading,
                "text": '\n'.join(current_content).strip()
            })
        
        # If no sections were detected, treat the entire text as one section
        if not sections:
            sections = [{"heading": "", "text": text}]
            
        return sections
    
    def entity_focused_summarize(
        self, 
        text: str, 
        entities: List[str] = None,
        method: str = "extractive",
        **kwargs
    ) -> Dict:
        """
        Create a summary focused on specific entities.
        
        Args:
            text: Input text to summarize
            entities: List of entities to focus on (auto-detected if None)
            method: Summarization method ('extractive' or 'abstractive')
            **kwargs: Additional parameters for the underlying summarization method
            
        Returns:
            Dictionary containing the summary and metadata
        """
        # This is a placeholder for a more sophisticated entity-focused summary
        # For now, we'll just boost sentences that contain the entities
        
        if entities is None:
            # In a real implementation, we would detect entities here
            # For now, we'll just extract proper nouns as a simple approximation
            words = nltk.word_tokenize(text)
            tagged = nltk.pos_tag(words)
            entities = list(set([word for word, tag in tagged if tag.startswith('NNP')]))
        
        # If no entities found or provided, fall back to regular summarization
        if not entities:
            if method == "extractive":
                return self.extractive_summarize(text, **kwargs)
            else:
                return self.abstractive_summarize(text, **kwargs)
        
        # For extractive summaries, we can boost entity-containing sentences
        if method == "extractive":
            sentences = nltk.sent_tokenize(text)
            
            # Score sentences based on entity presence
            entity_scores = []
            for sentence in sentences:
                score = 0
                for entity in entities:
                    if entity.lower() in sentence.lower():
                        score += 1
                entity_scores.append(score)
            
            # Normalize scores
            if max(entity_scores) > 0:
                entity_scores = [score / max(entity_scores) for score in entity_scores]
            
            # Get top sentences
            ratio = kwargs.get("ratio", 0.3)
            num_sentences = max(1, int(len(sentences) * ratio))
            
            # Combine with TF-IDF for better results
            tfidf_summary = self.extractive_summarize(text, **kwargs)
            tfidf_indices = tfidf_summary["selected_indices"]
            
            # Create a combined score
            combined_scores = []
            for i, (sentence, entity_score) in enumerate(zip(sentences, entity_scores)):
                tfidf_factor = 1.5 if i in tfidf_indices else 0.5
                combined_scores.append(entity_score * tfidf_factor)
            
            top_indices = np.argsort(combined_scores)[-num_sentences:]
            top_indices = sorted(top_indices)
            
            summary = " ".join([sentences[i] for i in top_indices])
            
            return {
                "summary": summary,
                "method": "entity_focused_extractive",
                "entities": entities,
                "sentence_count": len(top_indices),
                "selected_indices": top_indices.tolist() if isinstance(top_indices, np.ndarray) else top_indices
            }
        
        else:  # abstractive
            # For abstractive, we'll create a prompt that focuses on the entities
            entity_list = ", ".join(entities)
            prompt = f"Summarize the following text with focus on these entities: {entity_list}.\n\n{text}"
            
            # We'll use the base abstractive summarization
            result = self.abstractive_summarize(prompt, **kwargs)
            result["method"] = "entity_focused_abstractive"
            result["entities"] = entities
            
            return result