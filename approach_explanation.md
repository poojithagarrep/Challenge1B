# Intelligent Document Analyst System: Approach Explanation

Our Intelligent Document Analyst System is designed to extract and prioritize relevant information from a collection of PDF documents based on a user-defined persona and job-to-be-done. The system operates entirely on CPU, adheres to a strict memory footprint (≤ 1GB), and processes documents efficiently (≤ 60 seconds for 3-5 PDFs), without requiring internet access during execution.

## Methodology

The system is built upon a modular architecture, leveraging lightweight Natural Language Processing (NLP) techniques and heuristics to achieve its objectives:

1.  **Document Preprocessing (using `pdfplumber`):**
    * PDF documents are parsed using `pdfplumber`, which excels at extracting text along with crucial layout metadata like font size and position for each character. This rich data is fundamental for robust section identification.
    * Raw text is cleaned to remove extraneous whitespace and formatting artifacts.

2.  **Sectioning and Indexing:**
    * This is a critical step where documents are segmented into logical sections (e.g., "Introduction", "Methodology", "Conclusion").
    * **Heuristics-based Heading Detection:** We analyze font size (a line with significantly larger font than the page average is likely a heading), line length (headings are usually concise), and structural patterns (e.g., "1. Introduction", "II. Results"). A line must meet specific criteria (e.g., above average font size, limited word count if not a structured heading) to be classified as a section title.
    * Content following a detected heading is associated with that section until the next heading or end of the document. Each section retains its original document name, title, and starting page number.

3.  **Persona and Job-to-be-Done Understanding:**
    * Keywords are extracted from the persona's role description and the job-to-be-done using custom, regex-based tokenizers and a predefined, static list of English stop words. This avoids external NLTK data downloads and large model sizes.

4.  **Relevance Scoring and Prioritization (using TF-IDF & Heuristics):**
    * For each extracted section, a relevance score is calculated. This is a hybrid approach combining:
        * **TF-IDF Cosine Similarity:** Sections are vectorized using `TfidfVectorizer` (from `scikit-learn`), and their cosine similarity to a combined vector of the persona and job-to-be-done is computed. This quantifies semantic relevance.
        * **Keyword Matching:** A direct count of persona- and job-specific keywords within a section contributes to the score, with job-specific keywords weighted higher.
        * **Structural Heuristics:** Sections with common, inherently important titles (e.g., "Abstract", "Introduction", "Conclusion", "Methods", or domain-specific terms like "Cities", "Activities" for a travel planner) receive a significant score boost.
    * All sections are then ranked in descending order based on their combined relevance scores, assigning an `importance_rank` (1 being most relevant).

5.  **Sub-section Analysis (Extractive Summarization):**
    * For the top `N` (e.g., 5) most relevant sections, a "refined text" is generated.
    * An extractive summarization technique, a simplified version of the TextRank algorithm, is employed. Sentences within these key sections are ranked based on their similarity to other sentences in the section and their keyword density relative to the persona and job.
    * The top `K` (e.g., 3) highest-ranked sentences are extracted and concatenated to form the `refined_text`, providing a concise summary.

## Constraint Adherence

* **CPU-only & Model Size ≤ 1GB:** By avoiding large deep learning models and relying on `scikit-learn`'s `TfidfVectorizer` (which is small and efficient) and custom regex-based NLP components instead of NLTK's downloadable data, we maintain a minimal memory footprint and ensure CPU compatibility.
* **Processing Time ≤ 60 seconds:** `pdfplumber` is optimized for speed, and all subsequent NLP operations (TF-IDF, TextRank-like algorithms, custom tokenizers) are computationally lightweight, designed for rapid processing of document collections.
* **No Internet Access:** All dependencies (libraries and internal data like stop words) are bundled or pre-configured, requiring no external network calls during execution.