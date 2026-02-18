# ✅ Design Doc v2: The "Consensus Engine" 

| **Project** | Edu-Parser v2.0 (Adversarial Consensus & Fallback) |
| :--- | :--- |
| **Status** | **APPROVED for MVP** |
| **Core Philosophy** | "Cheap Consistency Check -> Expensive Fallback" |

## 1. Executive Summary
Instead of routing components, we process the **entire page** using two lightweight, architecturally distinct models (**HunyuanOCR** and **PaddleOCR-VL**). We compare their outputs mathematically. If they agree, we trust the result. If they disagree (high edit distance), we treat the page as "ambiguous" and fallback to a Frontier Model (Gemini 3 Pro or GPT-5.3 or Claude Opus 4.6) for resolution.

## 2. Architecture: The "Dual-Stream" Pipeline

### Stage 1: Parallel Inference (The "Cheap" Layer)
We utilize two SOTA small models identified in the 2025–2026 Survey. They are selected because they have **orthogonal failure modes** (Section 4.3 of Survey).

| Stream | Model | Type | Why this model? |
| :--- | :--- | :--- | :--- |
| **Stream A** | **HunyuanOCR (1B)** | **End-to-End VLM** | Excellent at raw text & formula generation (**94.73 CDM**). *Weakness:* Hallucinations in dense regions. |
| **Stream B** | **PaddleOCR-VL-1.5 (0.9B)** | **Cascaded Pipeline** | Excellent at Layout & Tables (**92.76 TEDS**) and "In-the-wild" distortions. *Weakness:* Error propagation. |

*   **Operation:** Both models ingest the *full page* at native resolution. No cropping. No routing.
*   **Latency:** Since both are ~1B params, they run in parallel on a single A100 (or split across 2x 4090s) in <1.5s per page.

### Stage 2: The Consensus Gate (The Logic Layer)
We do not use an LLM to check consistency (too slow). We use **Levenshtein Distance**.

1.  **Normalization:**
    *   Strip whitespace.
    *   Standardize LaTeX wrappers (convert `\[...\]` to `$$...$$`).
    *   Remove Markdown table syntax chars (`|`, `-`).
2.  **Comparison:**
    ```python
    similarity_score = Levenshtein.ratio(normalize(hunyuan_out), normalize(paddle_out))
    # Threshold derived from heuristics
    CONSENSUS_THRESHOLD = 0.92
    ```

### Stage 3: Conditional Fallback
*   **Case A: High Consensus (>0.92)**
    *   **Action:** Return **HunyuanOCR** output (generally cleaner LaTeX).
    *   **Cost:** ~$0.0005 per page (Energy cost).
    *   **Confidence:** High. If an E2E model and a Pipeline model agree, the text is almost certainly correct.

*   **Case B: Conflict (<0.92)**
    *   **Action:** The page is complex, degraded, or ambiguous.
    *   **Fallback:** Send the **original page image** (or specific bounding box of disagreement) to **Gemini 3 Pro**.
    *   **Prompt:** *"Extract the text/latex from this document perfectly. Return only Markdown."*
    *   **Cost:** ~$0.01 - $0.03 per page.
    *   **Frequency:** Expected to happen on ~10-15% of pages (Handwritten notes, old scans, complex tables).

## 3. Why This Wins (Technical Justification)

1.  **Solves the Hallucination vs. Propagation Dilemma:**
    *   The Survey (Section 4.3) notes that E2E models (Hunyuan) hallucinate content, while Pipeline models (Paddle) miss content. It is statistically improbable that Hunyuan will *hallucinate* the exact same string that Paddle *mis-detects*.
    *   Therefore, agreement = truth.

2.  **Operational Simplicity:**
    *   No complex cropping logic.
    *   No "Jagged Batches." We send full images to GPUs.
    *   We can run batch inference efficiently (Batch Size = 8 or 16) because every input is just "Full Page."

3.  **Cost Efficiency:**
    *   We effectively filter out 85% of "easy" pages (textbooks, clean papers) using cheap local compute.
    *   We only spend API budget on the "Hard 15%" where local models fail.

## 4. Implementation Details

**Hardware Stack:**
*   **Inference Server:** vLLM (supporting Hunyuan) + PaddlePaddle Runtime.
*   **Compute:** 1x NVIDIA A100 (80GB) or 2x RTX 4090 (24GB).

**Code Snippet (Logic Flow):**
```python
def process_page(image_path):
    # Parallel Inference
    future_a = executor.submit(hunyuan_inference, image_path)
    future_b = executor.submit(paddle_inference, image_path)
    
    res_a = future_a.result()
    res_b = future_b.result()

    # Consensus Check
    score = calculate_similarity(res_a, res_b)
    
    if score > 0.92:
        return res_a # Trust local
    else:
        # Log disagreement for future fine-tuning
        log_edge_case(image_path, res_a, res_b)
        # Call Frontier Model
        return gemini_flash_fallback(image_path) 
```

## 5. Risks & Mitigations

| Risk | Mitigation |
| :--- | :--- |
| **False Agreement** | Both models might agree on a wrong result if the image is extremely blurry. **Mitigation:** Unlikely given architectural differences, but acceptable for MVP. |
| **Latency on Fallback** | Calling Gemini adds ~2s latency. **Mitigation:** This only affects 15% of traffic. Acceptable trade-off for accuracy. |
| **Format Mismatch** | Hunyuan uses different LaTeX delimiters than Paddle. **Mitigation:** Aggressive Regex normalization before the Levenshtein check. |