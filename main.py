#  Intelligent Document Analyst System Code 

import pdfplumber
import re
import os
import json
from datetime import datetime
from collections import defaultdict, Counter
import math
import numpy as np

# For TF-IDF and Cosine Similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --- Custom Tokenizers  ---
def custom_sent_tokenize(text):
    """
    A simple regex-based sentence tokenizer.
    Splits by '.', '!', '?' followed by a space and an uppercase letter.
    """
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    final_sentences = []
    for s in sentences:
        final_sentences.extend(s.split('\n'))
    
    return [s.strip() for s in final_sentences if s.strip()]

def custom_word_tokenize(text):
    """
    A simple regex-based word tokenizer.
    Keeps only alphanumeric characters and converts to lowercase.
    """
    if not text:
        return []
    words = re.findall(r'\b\w+\b', text.lower())
    return words

# --- Custom Stop Words List ---
CUSTOM_STOP_WORDS = set([
    "a", "an", "the", "and", "but", "or", "as", "if", "then", "else", "when",
    "where", "why", "how", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "will", "would",
    "should", "can", "could", "may", "might", "must", "shall", "to", "of", "in",
    "on", "at", "by", "for", "with", "from", "up", "down", "out", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "s", "t", "can", "just", "don", "should", "now", "d", "ll",
    "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn",
    "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn",
    "wasn", "weren", "won", "wouldn", "about", "above", "across", "after", "afterwards",
    "against", "almost", "alone", "along", "already", "also", "although", "always",
    "among", "amongst", "amount", "and", "another", "anyhow", "anyone", "anything",
    "anywhere", "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "before", "beforehand", "behind", "being", "below",
    "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by",
    "call", "can", "cannot", "co", "con", "could", "couldnt", "cry", "de", "describe",
    "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either",
    "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every",
    "everyone", "everything", "everywhere", "except", "few", "ff", "fifth", "first",
    "five", "for", "former", "formerly", "forty", "found", "four", "from", "front",
    "full", "further", "get", "give", "go", "had", "has", "hasnt", "hence", "her",
    "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him",
    "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly",
    "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill",
    "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself",
    "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody",
    "none", "noone", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "s", "t", "can", "just", "don", "should", "now", "d", "ll",
    "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn",
    "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn",
    "wasn", "weren", "won", "wouldn", "about", "above", "across", "after", "afterwards",
    "against", "almost", "alone", "along", "already", "also", "although", "always",
    "among", "amongst", "amount", "and", "another", "anyhow", "anyone", "anything",
    "anywhere", "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "before", "beforehand", "behind", "being", "below",
    "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by",
    "call", "can", "cannot", "co", "con", "could", "couldnt", "cry", "de", "describe",
    "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either",
    "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every",
    "everyone", "everything", "everywhere", "except", "few", "ff", "fifth", "first",
    "five", "for", "former", "formerly", "forty", "found", "four", "from", "front",
    "full", "further", "get", "give", "go", "had", "has", "hasnt", "hence", "her",
    "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him",
    "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly",
    "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill",
    "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself",
    "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody",
    "none", "noone", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "s", "t", "can", "just", "don", "should", "now", "d", "ll",
    "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn",
    "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn",
    "wasn", "weren", "won", "wouldn", "about", "above", "across", "after", "afterwards",
    "against", "almost", "alone", "along", "already", "also", "although", "always",
    "among", "amongst", "amount", "and", "another", "anyhow", "anyone", "anything",
    "anywhere", "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "before", "beforehand", "behind", "being", "below",
    "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by",
    "call", "can", "cannot", "co", "con", "could", "couldnt", "cry", "de", "describe",
    "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either",
    "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every",
    "everyone", "everything", "everywhere", "except", "few", "ff", "fifth", "first",
    "five", "for", "former", "formerly", "forty", "found", "four", "from", "front",
    "full", "further", "get", "give", "go", "had", "has", "hasnt", "hence", "her",
    "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him",
    "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly",
    "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill",
    "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself",
    "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody",
    "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often",
    "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "overall", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming",
    "seems", "serious", "several", "she", "should", "show", "side", "since", "six",
    "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes",
    "somewhere", "still", "such", "system", "take", "ten", "th", "than", "that",
    "the", "their", "them", "themselves", "then", "thence", "there", "thereafter",
    "thereby", "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout", "thru",
    "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty",
    "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we",
    "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter",
    "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which",
    "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will",
    "with", "within", "without", "woman", "wonder", "would", "yet", "you", "your",
    "yours", "yourself", "yourselves", "z", "zero"
])



# Heuristics for heading detection using font size and structure
MIN_CHARS_IN_HEADING = 5    
MAX_WORDS_IN_NON_STRUCTURED_HEADING = 8 

# This defines the number of sections included in 'extracted_sections' and 'subsection_analysis'
TOP_N_RESULTS = 5 
NUM_SENTENCES_IN_REFINED_TEXT = 3


# --- Utility Functions ---
def clean_text(text):
    if text is None:
        return ""
    text = re.sub(r'[\s\n]+', ' ', text).strip()
    return text

def get_avg_font_size(chars):
    if not chars:
        return 0
    sizes = [char['size'] for char in chars if char['text'].strip() and char['size'] > 0]
    return sum(sizes) / len(sizes) if sizes else 0

def calculate_line_properties(line_chars):
    if not line_chars:
        return "", 0.0
    sorted_chars = sorted(line_chars, key=lambda x: x['x0'])
    line_text = "".join(c['text'] for c in sorted_chars).strip()
    line_sizes = [c['size'] for c in line_chars if c['text'].strip() and c['size'] > 0]
    avg_line_size = sum(line_sizes) / len(line_sizes) if line_sizes else 0.0
    return line_text, avg_line_size

# --- 1. Document Preprocessing Module ---
def extract_document_data(pdf_path):
    document_data = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                chars = page.chars
                
                line_char_map = defaultdict(list)
                for char in chars:
                    line_char_map[round(char['y0'])].append(char)
                
                sorted_line_keys = sorted(line_char_map.keys(), reverse=True) 

                page_lines_info = []
                for line_y in sorted_line_keys:
                    line_text, avg_line_size = calculate_line_properties(line_char_map[line_y])
                    if line_text:
                        page_lines_info.append({
                            "text": line_text,
                            "avg_font_size": avg_line_size,
                            "y0": line_y
                        })

                page_info = {
                    "page_number": page_num + 1,
                    "full_text": clean_text(page_text),
                    "lines": page_lines_info,
                    "page_avg_font_size": get_avg_font_size(chars)
                }
                document_data.append(page_info)
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None
    return document_data

# --- 2. Sectioning and Indexing Module ---
def identify_sections(document_data, doc_filename, config_params): # Accepts config_params now
    sections = []
    
    current_section = {
        "title": "Document Start",
        "raw_text_segments": [],
        "start_page": None,
        "end_page": None,
        "document": doc_filename
    }

    for page_data in document_data:
        page_num = page_data["page_number"]
        page_avg_font_size = page_data["page_avg_font_size"]
        
        if current_section["start_page"] is None:
            current_section["start_page"] = page_num
        current_section["end_page"] = page_num

        for line_info in page_data["lines"]:
            line_text = line_info["text"]
            avg_line_size = line_info["avg_font_size"]

            # Heuristic for a potential heading based on font size
            is_font_size_heading = (
                avg_line_size > page_avg_font_size * config_params["MIN_HEADING_SIZE_RATIO"] and
                len(custom_word_tokenize(line_text)) >= config_params["MIN_CHARS_IN_HEADING"] and
                not re.match(r'^\s*\d+\s*$', line_text) # Exclude lines that are just numbers (page numbers)
            )
            
            # Structured heading detection (e.g., "1. Introduction", "II. Methods")
            is_structured_heading = re.match(r'^\s*(\d+(\.\d+)*(\.\d+)*\s+)?([A-Z][a-zA-Z0-9\s-]+)$', line_text)
            
            # Combine heuristics:
            is_new_heading = False
            if is_structured_heading:
                is_new_heading = True
            elif is_font_size_heading and \
                 len(custom_word_tokenize(line_text)) <= config_params["MAX_WORDS_IN_NON_STRUCTURED_HEADING"] and \
                 not line_text.lower().startswith(("figure", "table", "appendix")): # Exclude common captions
                is_new_heading = True
            
            if is_new_heading:
                if current_section["raw_text_segments"]:
                    sections.append({
                        "document": current_section["document"],
                        "page_number": current_section['start_page'],
                        "section_title": current_section["title"],
                        "raw_text": clean_text(" ".join(current_section["raw_text_segments"])).strip(),
                        "importance_rank": -1
                    })
                
                current_section = {
                    "title": line_text,
                    "raw_text_segments": [],
                    "start_page": page_num,
                    "end_page": page_num,
                    "document": doc_filename
                }
            else:
                current_section["raw_text_segments"].append(line_text)
                current_section["end_page"] = page_num

    # Add the last section
    if current_section["raw_text_segments"] or current_section["title"] == "Document Start":
         sections.append({
            "document": current_section["document"],
            "page_number": current_section['start_page'],
            "section_title": current_section["title"],
            "raw_text": clean_text(" ".join(current_section["raw_text_segments"])).strip(),
            "importance_rank": -1
        })
    
    return sections

# --- 3. Persona and Job-to-be-Done Understanding Module ---
def get_keywords(text):
    """Extracts keywords from text, removing CUSTOM_STOP_WORDS and lowercasing."""
    words = custom_word_tokenize(text)
    keywords = [word for word in words if word.isalnum() and word not in CUSTOM_STOP_WORDS]
    return list(set(keywords))

# --- Document-Level Keyword Extraction (for internal use/logging) ---
def get_document_top_keywords(document_full_text, num_keywords=3):
    """
    Extracts top N TF-IDF keywords from a document's full text.
    """
    if not document_full_text:
        return []

    words = custom_word_tokenize(document_full_text)
    # Filter out stopwords and non-alphanumeric
    filtered_words = [word for word in words if word.isalnum() and word not in CUSTOM_STOP_WORDS]
    
    if not filtered_words:
        return []

    vectorizer = TfidfVectorizer(max_features=100) # Limit features for efficiency
    tfidf_matrix = vectorizer.fit_transform([" ".join(filtered_words)])

    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]

    # Get words and their scores, sort by score
    word_scores = list(zip(feature_names, tfidf_scores))
    word_scores.sort(key=lambda x: x[1], reverse=True)

    return [word for word, score in word_scores[:num_keywords]]


# --- 4. Relevance Scoring Module (using TF-IDF cosine similarity) ---
def rank_sections_tfidf(sections_list, persona_definition, job_to_be_done):
    context = f"{persona_definition}. {job_to_be_done}"
    
    all_texts = [sec["raw_text"] for sec in sections_list]
    vectorizer = TfidfVectorizer().fit(all_texts + [context])
    
    section_vectors = vectorizer.transform(all_texts)
    context_vector = vectorizer.transform([context])
    
    similarity_scores = cosine_similarity(section_vectors, context_vector).flatten()

    persona_keywords = get_keywords(persona_definition)
    job_keywords = get_keywords(job_to_be_done)

    for i, section in enumerate(sections_list):
        current_score = similarity_scores[i] * 100 # Scale to 0-100

        section_text_lower = section["raw_text"].lower()
        for keyword in job_keywords:
            if keyword in section_text_lower:
                current_score += 3
        for keyword in persona_keywords:
            if keyword in section_text_lower:
                current_score += 1

        common_high_relevance_terms = [
            "abstract", "introduction", "conclusion", "summary", "results", 
            "method", "methodology", "experiment", "analysis", "discussion",
            "financial", "revenue", "investment", "strategy", "overview",
            "concepts", "mechanisms", "kinetics", "preparation",
            # Add more travel-specific terms for better relevance
            "cities", "things to do", "cuisine", "restaurants", "hotels",
            "tips", "tricks", "traditions", "culture", "planning", "itinerary",
            "day trip", "activities", "entertainment", "transportation",
            "accommodation", "food", "drinks", "budget", "group", "friends",
            "college", "nightlife", "adventures", "packing", "packing tips",
            "coastal", "outdoor", "cultural experiences", "wine tasting",
            "family-friendly", "budget-friendly", "upscale", "luxurious",
            "history", "historical sites", "museums", "shopping", "markets",
            "festivals", "events", "spa", "wellness", "yoga", "meditation",
            "hiking", "biking", "water sports", "beach", "theme parks",
            "educational", "local experiences", "travel tips"
        ]
        for term in common_high_relevance_terms:
            if term in section["section_title"].lower() or term in section_text_lower[:200]: 
                current_score += 5

        section['relevance_score'] = current_score

    sections_list.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    for i, section in enumerate(sections_list):
        section['importance_rank'] = i + 1
    
    return sections_list

# --- 6. Sub-section Refinement Module (Extractive Summarization using a simplified TextRank idea) ---

def _get_sentence_vectors_simple(sentences):
    sentence_vectors = []
    for sentence in sentences:
        words = custom_word_tokenize(sentence)
        vector = Counter(words)
        sentence_vectors.append(vector)
    return sentence_vectors

def _cosine_similarity_from_counters(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[word] * vec2[word] for word in intersection])

    sum1 = sum([vec1[word]**2 for word in vec1.keys()])
    sum2 = sum([vec2[word]**2 for word in vec2.keys()])
    
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    
    if not denominator:
        return 0.0
    return numerator / denominator

def extractive_summarize_textrank_simplified(text, num_sentences, persona_keywords, job_keywords):
    sentences = custom_sent_tokenize(text)
    if not sentences:
        return ""
    
    sentences = [s for s in sentences if len(custom_word_tokenize(s)) > 5 and s.strip()]

    if not sentences:
        return ""

    sentence_vectors = _get_sentence_vectors_simple(sentences)

    num_sentences_actual = len(sentences)
    similarity_matrix = np.zeros((num_sentences_actual, num_sentences_actual))

    for i in range(num_sentences_actual):
        for j in range(num_sentences_actual):
            if i == j:
                continue
            similarity_matrix[i][j] = _cosine_similarity_from_counters(
                sentence_vectors[i], sentence_vectors[j]
            )

    row_sums = similarity_matrix.sum(axis=1, keepdims=True)
    similarity_matrix = np.where(row_sums == 0, 0, similarity_matrix / row_sums) 

    scores = np.ones(num_sentences_actual) / num_sentences_actual
    
    damping_factor = 0.85
    epsilon = 1e-4
    max_iterations = 100

    for _ in range(max_iterations):
        prev_scores = np.copy(scores)
        scores = (1 - damping_factor) + damping_factor * np.dot(similarity_matrix.T, scores)
        if np.linalg.norm(scores - prev_scores) < epsilon:
            break

    final_sentence_scores = []
    all_relevant_keywords = set(persona_keywords + job_keywords)
    for i, sentence in enumerate(sentences):
        keyword_count = sum(1 for kw in all_relevant_keywords if kw in sentence.lower())
        final_sentence_scores.append((scores[i] + keyword_count * 0.1, i, sentence))

    final_sentence_scores.sort(key=lambda x: x[0], reverse=True)
    
    selected_summary_sentences = []
    for score, original_idx, sentence_text in final_sentence_scores[:num_sentences]:
        selected_summary_sentences.append((original_idx, sentence_text))
    
    selected_summary_sentences.sort(key=lambda x: x[0])

    return " ".join([s[1] for s in selected_summary_sentences])

# --- Main System Orchestration ---
def intelligent_document_analyst( # Renamed back to intelligent_document_analyst for clarity
    document_paths,
    persona_definition,
    job_to_be_done,
    config=None,
    debug=False
):
    # Set config with defaults. Values from 'config' dict will override these.
    current_config = {
        "MIN_HEADING_SIZE_RATIO": MIN_HEADING_SIZE_RATIO,
        "MIN_CHARS_IN_HEADING": MIN_CHARS_IN_HEADING,
        "MAX_WORDS_IN_NON_STRUCTURED_HEADING": MAX_WORDS_IN_NON_STRUCTURED_HEADING,
        "TOP_N_RESULTS": TOP_N_RESULTS,
        "NUM_SENTENCES_IN_REFINED_TEXT": NUM_SENTENCES_IN_REFINED_TEXT
    }
    if config: # If user provides a config dict, update the current_config
        current_config.update(config)


    def log(msg):
        if debug:
            print("[DEBUG]", msg)

    all_extracted_sections_with_content = []
    sub_section_analysis_results = []

    persona_keywords = get_keywords(persona_definition)
    job_keywords = get_keywords(job_to_be_done)

    log(f"Persona Keywords: {persona_keywords}")
    log(f"Job Keywords: {job_keywords}")

    for doc_path in document_paths:
        doc_filename = os.path.basename(doc_path)
        log(f"\n--- Processing Document: {doc_filename} ---")

        document_data = extract_document_data(doc_path)
        if not document_data:
            log(f"Skipping {doc_filename} due to extraction failure.")
            continue

        # Get full text for document-level keyword extraction (for logging/debug)
        full_doc_text = " ".join([page['full_text'] for page in document_data])
        top_doc_keywords = get_document_top_keywords(full_doc_text, num_keywords=3)
        log(f"Top keywords for '{doc_filename}': {top_doc_keywords}")


        
        sections = identify_sections(document_data, doc_filename, config_params=current_config)
        log(f"Identified {len(sections)} sections in {doc_filename}")

        if not sections:
            log(f"No sections identified for {doc_filename}.")
            continue

        all_extracted_sections_with_content.extend(sections)

    if not all_extracted_sections_with_content:
        return json.dumps({
            "metadata": {
                "input_documents": [os.path.basename(p) for p in document_paths],
                "persona": persona_definition,
                "job_to_be_done": job_to_be_done,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": [],
            "subsection_analysis": [],
            "message": "No content extracted from documents."
        }, indent=2)

    # Rank sections
    ranked_sections = rank_sections_tfidf(all_extracted_sections_with_content, persona_definition, job_to_be_done)
    
    # --- Prepare the final output structure ---

    # 1. extracted_sections: Take only the TOP_N_RESULTS sections from the ranked list
    extracted_sections_output = []
    # Use slicing to get only the top N results, which is defined by current_config["TOP_N_RESULTS"]
    for section in ranked_sections[:current_config["TOP_N_RESULTS"]]:
        extracted_sections_output.append({
            "document": section['document'],
            "page_number": section['page_number'],
            "section_title": section['section_title'],
            "importance_rank": section['importance_rank']
        })

    # 2. subsection_analysis: Refined text for the TOP_N_RESULTS sections
    sub_section_analysis_results = []
    # Iterate through the same top N sections to generate refined text
    # Ensure they are sorted by importance_rank for consistent output
    top_n_for_refinement = sorted(ranked_sections[:current_config["TOP_N_RESULTS"]], key=lambda x: x['importance_rank'])

    for section in top_n_for_refinement:
        if section['raw_text']: # Ensure there's content to refine
            refined_text = extractive_summarize_textrank_simplified(
                section['raw_text'], current_config["NUM_SENTENCES_IN_REFINED_TEXT"], 
                persona_keywords, job_keywords
            )
            if refined_text:
                sub_section_analysis_results.append({
                    "document": section['document'],
                    "page_number": section['page_number'],
                    "refined_text": refined_text
                })
                
    # Final Output JSON structure
    output = {
        "metadata": {
            "input_documents": [os.path.basename(p) for p in document_paths],
            "persona": persona_definition,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": extracted_sections_output,
        "subsection_analysis": sub_section_analysis_results
    }
    
    return json.dumps(output, indent=2)
    