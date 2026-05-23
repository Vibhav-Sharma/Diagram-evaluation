import numpy as np
from typing import Tuple
from sentence_transformers import SentenceTransformer

# Load model globally to avoid reloading
_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def evaluate_semantic_coherence(text_a: str, text_b: str) -> Tuple[float, float, str]:
    """
    Evaluates semantic coherence, phrase continuity, and instruction plausibility.
    Returns:
        semantic_score (float): 0.0 to 1.0 (overall confidence in the merge)
        contextual_similarity (float): 0.0 to 1.0 (similarity of parts to the whole)
        reasoning (str): Textual explanation of the score
    """
    if not text_a or not text_b:
        return 0.0, 0.0, "Empty text"
        
    a_up = text_a.strip().upper()
    b_up = text_b.strip().upper()
    
    # 1. Procedural Action heuristics
    action_verbs = {"START", "STOP", "END", "BUY", "SELL", "PROCESS", "WAIT", "ADD", "REMOVE", "CHECK", "IF", "THEN", "READ", "WRITE", "OUTPUT", "INPUT", "PRINT"}
    a_words = a_up.split()
    b_words = b_up.split()
    
    a_has_action = any(w in action_verbs for w in a_words)
    b_has_action = any(w in action_verbs for w in b_words)
    
    # Connectors should not be merged into long phrases usually, but maybe they are parts of "WAIT FOR" -> "WATER TO BOIL"
    
    # Cosine Similarity between A, B and AB
    model = get_model()
    embeddings = model.encode([text_a, text_b, f"{text_a} {text_b}"])
    
    norm_a = np.linalg.norm(embeddings[0])
    norm_b = np.linalg.norm(embeddings[1])
    norm_ab = np.linalg.norm(embeddings[2])
    
    sim_ab = 0.0
    if norm_a > 0 and norm_b > 0:
        sim_ab = np.dot(embeddings[0], embeddings[1]) / (norm_a * norm_b)
        
    sim_combined_a = 0.0
    sim_combined_b = 0.0
    if norm_a > 0 and norm_ab > 0:
        sim_combined_a = np.dot(embeddings[0], embeddings[2]) / (norm_a * norm_ab)
    if norm_b > 0 and norm_ab > 0:
        sim_combined_b = np.dot(embeddings[1], embeddings[2]) / (norm_b * norm_ab)
        
    contextual_similarity = float((sim_combined_a + sim_combined_b) / 2.0)
    
    # Score logic - Context-First Grouping
    semantic_score = 0.0
    reasoning = ""
    
    # Instruction plausibility
    if a_has_action and not b_has_action:
        semantic_score += 0.4
        reasoning += "Action continuation. "
    elif not a_has_action and not b_has_action:
        semantic_score += 0.2
        reasoning += "Noun phrase continuation. "
        
    # Phrase continuity
    if contextual_similarity > 0.85:
        semantic_score += 0.5
        reasoning += "High phrase continuity. "
    elif contextual_similarity > 0.7:
        semantic_score += 0.3
        reasoning += "Moderate phrase continuity. "
        
    # Semantic similarity
    if sim_ab > 0.4:
        semantic_score += 0.2
        reasoning += "High semantic similarity. "
        
    # Penalties
    if a_has_action and b_has_action and sim_ab < 0.3:
        semantic_score -= 0.3
        reasoning += "Disjoint actions. "
        
    # Cap between 0.0 and 1.0
    semantic_score = max(0.0, min(1.0, semantic_score))
    
    if semantic_score >= 0.5:
        reasoning = f"Coherent ({semantic_score:.2f}): " + reasoning
    else:
        reasoning = f"Incoherent ({semantic_score:.2f}): " + reasoning
        
    return semantic_score, float(contextual_similarity), reasoning.strip()

def invoke_vlm_arbitrator(image_crop, text_a: str, text_b: str) -> bool:
    """
    Placeholder for Florence-2 validation.
    Called when semantic_score is uncertain (e.g. 0.4 to 0.6).
    """
    return False
