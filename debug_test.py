#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import logging
from difflib import SequenceMatcher

# Setup logging with UTF-8
logging.basicConfig(filename='debug.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')

def exact_question_check(query, json_files):
    """So sánh query với các câu hỏi trong JSON"""
    query = query.lower().strip()
    logging.info(f"Checking exact match for query: {query}")
    
    best_match = None
    best_similarity = 0
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data['data']:
                    for paragraph in item['paragraphs']:
                        for qa in paragraph['qas']:
                            question = qa['question'].lower().strip()
                            
                            # Kiểm tra exact match
                            if query == question:
                                answer = qa['answers'][0]['text']
                                logging.info(f"Exact match found: {question}")
                                return answer
                            
                            # Kiểm tra similarity
                            similarity = SequenceMatcher(None, query, question).ratio()
                            if similarity > best_similarity and similarity > 0.7:
                                best_similarity = similarity
                                best_match = {
                                    'answer': qa['answers'][0]['text'],
                                    'question': question,
                                    'similarity': similarity
                                }
                                logging.info(f"New best match: {question} (similarity: {similarity:.3f})")
        except Exception as e:
            logging.error(f"Error reading {json_file}: {e}")
    
    if best_match:
        logging.info(f"Best match found: {best_match['question']} (similarity: {best_match['similarity']:.3f})")
        return best_match['answer']
    
    logging.info("No exact match found")
    return None

if __name__ == "__main__":
    # Test với file data_chuDe_2.json
    json_files = [r"D:\du_an_chatBot_noi_bo\data\data_chuDe_2.json"]
    
    query = "Cách đóng gói Hàng giá trị cao?"
    answer = exact_question_check(query, json_files)
    
    
    try:
        print(f"Query: {query}")
        print(f"Answer: {answer}")
    except UnicodeEncodeError:
        with open('debug_result.txt', 'w', encoding='utf-8') as f:
            f.write(f"Query: {query}\n")
            f.write(f"Answer: {answer}\n")
        print("Results saved to debug_result.txt")
