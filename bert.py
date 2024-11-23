import re
import nltk
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os  

nltk.download('punkt')

unwanted_keywords = [
    "subscribe", "like the video", "follow me", "check out my channel",
    "you know", "basically", "okay", "thank you for watching", 
    "see you in the next video", "click the link", "smash the like button",
    "quick video", "I'm not sure", "just wanted to"
]

def preprocess(text):
    return text.lower()

def filter_unwanted_sentences(transcript, unwanted_keywords):
    filtered_transcript = []
    
    for sentence in transcript:
        processed_sentence = preprocess(sentence)
        
        if not any(re.search(rf'\b{keyword}\b', processed_sentence) for keyword in unwanted_keywords):
            filtered_transcript.append(sentence)
    
    return filtered_transcript

def split_into_sentences(text):
    return nltk.sent_tokenize(text)

def identify_outliers(similarity_matrix, threshold=0.5):
    similarity_matrix = np.array(similarity_matrix)
    
    outlier_indices = []
    for i, row in enumerate(similarity_matrix):
        avg_similarity = np.mean(row)
        if avg_similarity < threshold:
            outlier_indices.append(i)
    
    return outlier_indices

def process_transcript(input_file, output_file):
    with open(input_file, 'r') as file:
        transcript_text = file.read()
    
    sentences = split_into_sentences(transcript_text)
    
    filtered_transcript = filter_unwanted_sentences(sentences, unwanted_keywords)
    
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    
    embeddings = model.encode(filtered_transcript, convert_to_tensor=True)
    
    similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings)
    
    outlier_indices = identify_outliers(similarity_matrix, threshold=0.5)
    
    final_transcript = [sentence for i, sentence in enumerate(filtered_transcript) if i not in outlier_indices]
    
    with open(output_file, 'w') as file:
        for sentence in final_transcript:
            file.write(sentence + '\n')

    print(f"Filtered transcript has been saved to {output_file}")

def process_multiple_transcripts(input_files, output_files):
    if len(input_files) != len(output_files):
        raise ValueError("The number of input files and output files must be the same.")
    
    for input_file, output_file in zip(input_files, output_files):
        process_transcript(input_file, output_file)

input_files = ['transcript1.txt', 'transcript2.txt', 'transcript3.txt', 'transcript4.txt', 'transcript5.txt', 'transcript6.txt', 'transcript7.txt', 'transcript8.txt', 'transcript9.txt', 'transcript10.txt']  # List of input files
output_files = ['output_transcript1.txt', 'output_transcript2.txt', 'output_transcript3.txt', 'output_transcript4.txt', 'output_transcript5.txt', 'output_transcript6.txt', 'output_transcript7.txt', 'output_transcript8.txt', 'output_transcript9.txt', 'output_transcript10.txt']  # List of corresponding output files

process_multiple_transcripts(input_files, output_files)
