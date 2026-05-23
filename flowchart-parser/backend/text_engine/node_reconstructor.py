import cv2
import numpy as np
import spacy
from text_engine.semantic_scorer import get_model # Reuse SentenceTransformers model

# Load spacy model globally
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None # Handle fallback if not loaded yet

def detect_shapes(image: np.ndarray) -> list:
    """
    Detect structural flowchart containers (rectangles, diamonds, ovals).
    Returns list of dicts: { "id": str, "type": str, "polygon": np.ndarray, "bbox": [x, y, w, h] }
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate edges to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shapes = []
    shape_idx = 1
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500: # Filter noise
            continue
            
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        
        shape_type = "unknown"
        if len(approx) == 4:
            aspect_ratio = float(w) / h
            if 0.85 <= aspect_ratio <= 1.15:
                # Might be diamond if rotated, or square. We can check angles, but let's keep it simple
                shape_type = "decision" # Diamonds are often decision nodes
            else:
                shape_type = "process" # Rectangles
        elif len(approx) > 4:
            # Circle or oval
            shape_type = "terminal"
            
        shapes.append({
            "id": f"shape_{shape_idx}",
            "type": shape_type,
            "polygon": cnt, # Use raw contour for precise point-in-polygon test
            "bbox": [x, y, w, h]
        })
        shape_idx += 1
        
    return shapes

def detect_arrows(image: np.ndarray) -> list:
    """
    Detect arrows (shafts and heads).
    Returns list of dicts: { "id": str, "start_point": (x, y), "end_point": (x, y), "label": None }
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=40, maxLineGap=10)
    
    arrows = []
    if lines is not None:
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            arrows.append({
                "id": f"arrow_{i+1}",
                "start_point": (x1, y1),
                "end_point": (x2, y2),
                "label": None # Populated later in reconstruct_graph if needed
            })
            
    return arrows

def assign_fragments_to_shapes(ocr_results: list, shapes: list) -> tuple:
    """
    Assign OCR text blocks to shapes based on their centroids.
    Returns: (assigned_fragments, orphans)
    """
    assigned_fragments = {s["id"]: [] for s in shapes}
    orphans = []
    
    for block in ocr_results:
        # Assuming block format has bbox [x_min, y_min, x_max, y_max] or center
        if "center" in block and isinstance(block["center"], dict):
            cx = block["center"]["x"]
            cy = block["center"]["y"]
        elif "bbox" in block:
            cx = (block["bbox"][0] + block["bbox"][2]) / 2.0
            cy = (block["bbox"][1] + block["bbox"][3]) / 2.0
        else:
            # Fallback
            cx = (block[0][0][0] + block[0][2][0]) / 2.0
            cy = (block[0][0][1] + block[0][2][1]) / 2.0
            
        assigned = False
        for s in shapes:
            # Check if centroid is inside contour
            dist = cv2.pointPolygonTest(s["polygon"], (cx, cy), False)
            if dist >= 0: # inside or on edge
                assigned_fragments[s["id"]].append(block)
                assigned = True
                break
                
        if not assigned:
            orphans.append(block)
            
    return assigned_fragments, orphans

def assemble_node_text(assigned_fragments: list) -> str:
    """
    Sorts fragments top-to-bottom, left-to-right within y-bands, and concatenates.
    """
    if not assigned_fragments:
        return ""
        
    def get_y(b):
        if "center" in b and isinstance(b["center"], dict): return b["center"]["y"]
        elif "bbox" in b: return (b["bbox"][1] + b["bbox"][3]) / 2.0
        return (b[0][0][1] + b[0][2][1]) / 2.0
        
    def get_x(b):
        if "center" in b and isinstance(b["center"], dict): return b["center"]["x"]
        elif "bbox" in b: return (b["bbox"][0] + b["bbox"][2]) / 2.0
        return (b[0][0][0] + b[0][2][0]) / 2.0
        
    def get_text(b):
        if "text" in b: return b["text"]
        return b[1]
        
    # Estimate median height for y-band tolerance
    heights = []
    for b in assigned_fragments:
        if "bbox" in b: heights.append(b["bbox"][3] - b["bbox"][1])
        else: heights.append(b[0][2][1] - b[0][0][1])
        
    median_h = np.median(heights) if heights else 20.0
    tolerance = 1.5 * median_h
    
    sorted_by_y = sorted(assigned_fragments, key=get_y)
    
    bands = []
    current_band = []
    current_y = None
    
    for b in sorted_by_y:
        y = get_y(b)
        if current_y is None:
            current_y = y
            current_band.append(b)
        elif abs(y - current_y) <= tolerance:
            current_band.append(b)
            # Update running average of y
            current_y = sum(get_y(x) for x in current_band) / len(current_band)
        else:
            bands.append(sorted(current_band, key=get_x))
            current_band = [b]
            current_y = y
            
    if current_band:
        bands.append(sorted(current_band, key=get_x))
        
    final_text = []
    for band in bands:
        for b in band:
            final_text.append(get_text(b))
            
    return " ".join(final_text)

def _is_complete_instruction(phrase: str) -> bool:
    """Uses rules and spaCy to check if a phrase is a complete flowchart step."""
    if nlp is None:
        return False
        
    phrase = phrase.strip().upper()
    standalone = {"START", "END", "YES", "NO", "STOP"}
    if phrase in standalone:
        return True
        
    if phrase.endswith("?"):
        return True
        
    doc = nlp(phrase.lower())
    
    has_verb = any(token.pos_ == "VERB" for token in doc)
    has_obj = any(token.dep_ in {"dobj", "pobj", "attr"} for token in doc)
    
    word_count = len([t for t in doc if not t.is_punct])
    
    if word_count >= 6 and has_verb:
        return True
        
    if has_verb and has_obj:
        return True
        
    # Incomplete markers
    incomplete_starters = {"is", "are", "do", "does"}
    if doc and doc[0].text in incomplete_starters and not has_obj:
        return False
        
    prepositions = {"AND", "TO", "THE", "FOR"}
    if phrase in prepositions:
        return False
        
    return False

def _line_intersects_segment(line_p1, line_p2, seg_p1, seg_p2):
    """Check if arrow line intersects the line segment between two centroids."""
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    
    return ccw(line_p1, seg_p1, seg_p2) != ccw(line_p2, seg_p1, seg_p2) and ccw(line_p1, line_p2, seg_p1) != ccw(line_p1, line_p2, seg_p2)

def recover_orphans(orphans: list, arrows: list) -> list:
    """
    Recovers orphan text fragments into nodes using a local neighborhood graph
    and spaCy-based completeness checks. Hard blocks merges across arrows.
    """
    if not orphans:
        return []
        
    def get_center(b):
        if "center" in b and isinstance(b["center"], dict): return (b["center"]["x"], b["center"]["y"])
        elif "bbox" in b: return ((b["bbox"][0] + b["bbox"][2]) / 2.0, (b["bbox"][1] + b["bbox"][3]) / 2.0)
        return ((b[0][0][0] + b[0][2][0]) / 2.0, (b[0][0][1] + b[0][2][1]) / 2.0)
        
    def get_text(b):
        if "text" in b: return b["text"]
        return b[1]

    # Calculate median local text density (distance to nearest neighbor) for normalization
    centers = [get_center(b) for b in orphans]
    if len(centers) > 1:
        import scipy.spatial
        tree = scipy.spatial.KDTree(centers)
        dists, _ = tree.query(centers, k=2)
        median_dist = np.median(dists[:, 1])
    else:
        median_dist = 50.0
        
    norm_factor = median_dist if median_dist > 0 else 50.0
    
    import networkx as nx
    G = nx.Graph()
    for i, b in enumerate(orphans):
        G.add_node(i, data=b, center=centers[i], text=get_text(b))
        
    for i in range(len(orphans)):
        for j in range(i + 1, len(orphans)):
            c1 = centers[i]
            c2 = centers[j]
            dist = np.hypot(c1[0] - c2[0], c1[1] - c2[1])
            
            # Check hard cut
            crossed = False
            for arr in arrows:
                if _line_intersects_segment(arr["start_point"], arr["end_point"], c1, c2):
                    crossed = True
                    break
                    
            if crossed:
                continue
                
            # Base weight
            normalized_dist = dist / norm_factor
            if normalized_dist == 0: normalized_dist = 0.001
            weight = 1.0 / normalized_dist
            
            # Bonus for alignment
            if abs(c1[0] - c2[0]) < 15 or abs(c1[1] - c2[1]) < 15:
                weight += 0.2
                
            if weight > 0.5: # Threshold to connect
                G.add_edge(i, j, weight=weight)
                
    # Traverse and stop merging when complete
    visited = set()
    recovered_nodes = []
    
    for node in G.nodes():
        if node in visited:
            continue
            
        group_indices = [node]
        visited.add(node)
        
        # Greedy expansion
        while True:
            current_text = assemble_node_text([orphans[idx] for idx in group_indices])
            if _is_complete_instruction(current_text):
                break
                
            # Find best unvisited neighbor
            neighbors = [(n, G[n_in][n]['weight']) for n_in in group_indices for n in G.neighbors(n_in) if n not in visited]
            if not neighbors:
                break
                
            best_neighbor = max(neighbors, key=lambda x: x[1])[0]
            group_indices.append(best_neighbor)
            visited.add(best_neighbor)
            
        recovered_nodes.append([orphans[idx] for idx in group_indices])
        
    return recovered_nodes

def reconstruct_graph(nodes: list, arrows: list) -> dict:
    """Connects nodes using arrow endpoints."""
    edges = []
    
    def get_node_center(n):
        return ((n["bounding_box"][0] + n["bounding_box"][2]) / 2.0, (n["bounding_box"][1] + n["bounding_box"][3]) / 2.0)
        
    for arr in arrows:
        start_pt = arr["start_point"]
        end_pt = arr["end_point"]
        
        best_from = None
        min_from_dist = float('inf')
        
        best_to = None
        min_to_dist = float('inf')
        
        for n in nodes:
            nc = get_node_center(n)
            d_start = np.hypot(nc[0] - start_pt[0], nc[1] - start_pt[1])
            d_end = np.hypot(nc[0] - end_pt[0], nc[1] - end_pt[1])
            
            if d_start < min_from_dist:
                min_from_dist = d_start
                best_from = n["id"]
                
            if d_end < min_to_dist:
                min_to_dist = d_end
                best_to = n["id"]
                
        if best_from and best_to:
            edges.append({
                "id": arr["id"],
                "from": best_from,
                "to": best_to,
                "label": arr["label"]
            })
            
    return {"edges": edges}

def reconstruct_nodes(image: np.ndarray, ocr_results: list) -> dict:
    """Orchestrates the entire container-first pipeline."""
    warnings = []
    
    shapes = detect_shapes(image)
    arrows = detect_arrows(image)
    
    assigned, orphans = assign_fragments_to_shapes(ocr_results, shapes)
    
    nodes = []
    node_idx = 1
    
    # Process container nodes
    for s in shapes:
        fragments = assigned.get(s["id"], [])
        if fragments:
            text = assemble_node_text(fragments)
            nodes.append({
                "id": f"node_{node_idx}",
                "type": s["type"],
                "text": text,
                "bounding_box": s["bbox"],
                "source": "contained"
            })
            node_idx += 1
            
            # Semantic coherence validation
            model = get_model()
            if model and len(fragments) > 1:
                sub_texts = [f.get("text", f[1]) for f in fragments]
                embeddings = model.encode(sub_texts)
                sims = []
                for i in range(len(embeddings)-1):
                    sim = np.dot(embeddings[i], embeddings[i+1]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1]))
                    sims.append(sim)
                if sims and np.mean(sims) < 0.2:
                    warnings.append(f"node_{node_idx-1} text has low internal coherence — review suggested")
            
    # Process orphan nodes
    recovered = recover_orphans(orphans, arrows)
    for group in recovered:
        text = assemble_node_text(group)
        
        # Calculate group bbox
        x_mins, y_mins, x_maxs, y_maxs = [], [], [], []
        for b in group:
            if "bbox" in b:
                x_mins.append(b["bbox"][0])
                y_mins.append(b["bbox"][1])
                x_maxs.append(b["bbox"][2])
                y_maxs.append(b["bbox"][3])
            else:
                x_mins.append(b[0][0][0])
                y_mins.append(b[0][0][1])
                x_maxs.append(b[0][2][0])
                y_maxs.append(b[0][2][1])
                
        bbox = [min(x_mins), min(y_mins), max(x_maxs) - min(x_mins), max(y_maxs) - min(y_mins)]
        
        nodes.append({
            "id": f"node_{node_idx}",
            "type": "unknown",
            "text": text,
            "bounding_box": bbox,
            "source": "orphan_recovered"
        })
        node_idx += 1
        
    graph = reconstruct_graph(nodes, arrows)
    
    return {
        "nodes": nodes,
        "edges": graph["edges"],
        "warnings": warnings
    }
