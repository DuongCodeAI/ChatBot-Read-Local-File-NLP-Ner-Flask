import json
import os
import logging
from difflib import SequenceMatcher

# Setup logging
logging.basicConfig(filename='qa_pipeline.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')

def exact_question_check(query, json_files):
    """
    So sánh query với các câu hỏi trong JSON. Nếu match, trả câu trả lời.
    Input: query (str), json_files (list of file paths)
    Output: answer (str) nếu match, None nếu không
    """
    query = query.lower().strip()  # Chuẩn hóa query
    logging.info(f"Checking exact match for query: {query}")
    
    best_match = None
    best_similarity = 0
    
    # Tạo keywords từ query để matching tốt hơn
    query_keywords = set(preprocess_query(query).split())
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data['data']:
                    for paragraph in item['paragraphs']:
                        for qa in paragraph['qas']:
                            question = qa['question'].lower().strip()
                            
                            # Kiểm tra exact match trước
                            if query == question:
                                answer = qa['answers'][0]['text']
                                logging.info(f"Exact match found in {json_file}, question: {question}, answer: {answer}")
                                return answer
                            
                            # Kiểm tra keyword overlap
                            question_keywords = set(preprocess_query(question).split())
                            keyword_overlap = len(query_keywords.intersection(question_keywords))
                            
                            # Kiểm tra similarity cao
                            similarity = SequenceMatcher(None, query, question).ratio()
                            
                            # Tính combined score
                            combined_score = similarity + 0.1 * keyword_overlap
                            
                            if combined_score > best_similarity and combined_score > 0.7:  # Giảm ngưỡng và thêm keyword boost
                                best_similarity = combined_score
                                best_match = {
                                    'answer': qa['answers'][0]['text'],
                                    'question': question,
                                    'file': json_file,
                                    'similarity': combined_score
                                }
        except Exception as e:
            logging.error(f"Error reading {json_file}: {e}")
    
    # Trả về best match nếu có
    if best_match and best_similarity > 0.7:
        logging.info(f"Best match found: {best_match['question']} (similarity: {best_similarity:.3f})")
        return best_match['answer']
    
    logging.info("No exact match found, proceeding to next steps")
    return None 
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from underthesea import word_tokenize

def preprocess_text(text):
    """Chuẩn hóa văn bản: lowercase, tokenize bằng Underthesea, bỏ dấu câu"""
    text = text.lower().strip()
    tokens = word_tokenize(text, format='text')  # Underthesea tokenize
    return tokens

def load_data(json_files, cache_path='vectors.pkl'):
    """
    Load JSON files, index questions, and cache TF-IDF vectors
    Input: json_files (list of file paths), cache_path (str)
    Output: data (list of dicts), vectorizer, doc_vectors
    """
    data = []
    questions = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                for item in json_data['data']:
                    for paragraph in item['paragraphs']:
                        for qa in paragraph['qas']:
                            data.append({
                                'file': json_file,
                                'question': qa['question'],
                                'answer': qa['answers'][0]['text'],
                                'context': paragraph['context']
                            })
                            questions.append(preprocess_text(qa['question']))
            logging.info(f"Loaded {json_file} with {len(json_data['data'])} FAQs")
        except Exception as e:
            logging.error(f"Error loading {json_file}: {e}")
            
    
    # Vectorize questions
    try:
        vectorizer = TfidfVectorizer()
        doc_vectors = vectorizer.fit_transform(questions)
        with open(cache_path, 'wb') as f:
            pickle.dump((vectorizer, doc_vectors), f)
        logging.info(f"Cached TF-IDF vectors to {cache_path}")
    except Exception as e:
        logging.error(f"Error vectorizing or caching: {e}")
        return data, None, None
    
    return data, vectorizer, doc_vectors
def load_stopwords():
    """Tải danh sách stop words tiếng Việt (có thể dùng file hoặc danh sách mặc định)"""
    # Danh sách stopwords tối thiểu để giữ lại các từ quan trọng
    stopwords = {'là', 'của', 'và', 'trong', 'cho', 'được', 'có', 'một', 'các', 'những', 'này', 'đó', 'nào', 'gì', 'khi', 'nếu', 'để', 'với', 'từ', 'đến', 'tại', 'về', 'theo', 'như', 'hoặc', 'cũng', 'đã', 'sẽ', 'đang', 'được', 'có thể', 'phải', 'cần', 'nên', 'thì', 'mà', 'để', 'cho', 'về', 'của', 'trong', 'với', 'từ', 'đến', 'tại', 'theo', 'như', 'hoặc', 'cũng', 'đã', 'sẽ', 'đang', 'được', 'có thể', 'phải', 'cần', 'nên', 'thì', 'mà'}
    return stopwords

def preprocess_query(query):
    """
    Preprocess query: lowercase, tokenize, remove punctuation, stop words
    Input: query (str)
    Output: processed query (str)
    """
    stopwords = load_stopwords()
    
    # Lowercase and tokenize
    query = query.lower().strip()
    tokens = word_tokenize(query, format='text')
    
    # Remove punctuation and stop words, but keep important words
    tokens = [word for word in tokens.split() if word not in stopwords and len(word) > 1]
    processed_query = ' '.join(tokens)
    
    logging.info(f"Preprocessed query: {processed_query}")
    return processed_query

from underthesea import ner

def map_ner_to_custom(entity, entity_type):
    """Map nhãn Underthesea sang thực thể tùy chỉnh"""
    # Xử lý encoding issues
    entity = entity.strip()
    if not entity or len(entity) < 2:
        return None
    
    
    if entity_type.startswith('B-') or entity_type.startswith('I-'):
        if 'PER' in entity_type:  # Person
            return 'Person'
        elif 'LOC' in entity_type:  # Location
            return 'Location'
        elif 'ORG' in entity_type:  # Organization
            return 'Organization'
        elif 'MISC' in entity_type:  # Miscellaneous
            return 'Misc'
    
    # Heuristic mapping dựa trên nội dung
    entity_lower = entity.lower()
    if entity.isnumeric() or 'triệu' in entity_lower or 'nghìn' in entity_lower:
        return 'Amount'
    elif any(word in entity_lower for word in ['bảo hành', 'dịch vụ', 'hỗ trợ']):
        return 'Service'
    elif any(word in entity_lower for word in ['hàng', 'sản phẩm', 'máy', 'điện thoại', 'laptop']):
        return 'Product'
    elif any(word in entity_lower for word in ['giá trị', 'cao', 'quan trọng']):
        return 'Value'
    
    return None

def extract_keywords_from_query(query):
    """
    Trích xuất keywords quan trọng từ query
    Input: query (str)
    Output: list of (keyword, type)
    """
    keywords = []
    query_lower = query.lower()
    
    # Các từ khóa quan trọng
    important_phrases = [
        ('hàng giá trị cao', 'Product'),
        ('đóng gói', 'Action'),
        ('vận chuyển', 'Action'),
        ('bảo hiểm', 'Service'),
        ('thời gian', 'Time'),
        ('chi phí', 'Cost'),
        ('giờ làm việc', 'Time'),
        ('đồng phục', 'Policy'),
        ('nghỉ phép', 'Policy'),
        ('lương', 'Policy')
    ]
    
    for phrase, phrase_type in important_phrases:
        if phrase in query_lower:
            keywords.append((phrase, phrase_type))
    
    return keywords

def ner_extraction(query):
    """
    Trích xuất thực thể từ query
    Input: query (str)
    Output: list of (entity, custom_type)
    """
    # Thử NER trước
    try:
        entities = ner(query)  # Underthesea NER
        custom_entities = []
        
        for entity, entity_type, _, _ in entities:
            mapped_type = map_ner_to_custom(entity, entity_type)
            if mapped_type:
                custom_entities.append((entity, mapped_type))
    except Exception as e:
        logging.warning(f"NER failed: {e}")
        custom_entities = []
    
    # Thêm keyword extraction
    keyword_entities = extract_keywords_from_query(query)
    custom_entities.extend(keyword_entities)
    
    logging.info(f"NER entities: {custom_entities}")
    return custom_entities

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def semantic_search(query, data, vectorizer, doc_vectors, entities, top_k=5):
    """
    Tìm FAQ tương tự nhất bằng TF-IDF và cosine similarity, boost entity
    Input: query (str), data (list), vectorizer, doc_vectors, entities (list), top_k (int)
    Output: list of (index, score)
    """
    processed_query = preprocess_query(query)
    query_vec = vectorizer.transform([processed_query])
    
    # Cosine similarity
    similarities = cosine_similarity(query_vec, doc_vectors)[0]
    
    # Entity boosting
    for entity, _ in entities:
        if entity in vectorizer.vocabulary_:
            idx = vectorizer.vocabulary_[entity]
            similarities += 0.1 * (doc_vectors[:, idx].toarray().flatten())  # Boost entity
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = [(idx, similarities[idx]) for idx in top_indices]
    
    logging.info(f"Semantic search results: {results}")
    return results

def rule_based_matching(query, data, top_results, entities, threshold=0.3):
    """
    Lọc FAQ dựa trên entity và keyword, kiểm tra ngưỡng
    Input: query (str), data (list), top_results (list of (idx, score)), entities (list), threshold (float)
    Output: best index (int) hoặc None nếu fallback
    """
    keywords = preprocess_query(query).split()
    best_idx, best_score = None, 0
    query_lower = query.lower()
    
    for idx, score in top_results:
        if score < threshold:
            continue
        question = data[idx]['question'].lower()
        context = data[idx]['context'].lower()
        
        # Entity matching với trọng số cao
        entity_score = 0
        for entity, entity_type in entities:
            entity_lower = entity.lower()
            if entity_lower in question or entity_lower in context:
                # Boost score dựa trên loại entity
                if entity_type in ['Product', 'Action', 'Service']:
                    entity_score += 0.3
                else:
                    entity_score += 0.1
        
        # Keyword matching
        keyword_count = sum(1 for kw in keywords if kw in question or kw in context)
        keyword_score = 0.1 * keyword_count
        
        # Exact phrase matching (quan trọng nhất)
        phrase_score = 0
        if len(keywords) >= 2:
            phrase = ' '.join(keywords)
            if phrase in question or phrase in context:
                phrase_score = 0.5
        
        # Special phrase matching cho các cụm từ quan trọng
        special_phrases = ['hàng giá trị cao', 'đóng gói', 'giờ làm việc', 'đồng phục']
        special_score = 0
        for phrase in special_phrases:
            if phrase in query_lower and phrase in (question + ' ' + context):
                special_score += 0.4
        
        # Combined score
        combined_score = score + entity_score + keyword_score + phrase_score + special_score
        
        if combined_score > best_score:
            best_idx, best_score = idx, combined_score
            logging.info(f"New best match: idx={idx}, score={combined_score:.3f}, entity_score={entity_score:.3f}, keyword_score={keyword_score:.3f}, phrase_score={phrase_score:.3f}, special_score={special_score:.3f}")
    
    if best_idx is None:
        logging.warning("No match found, falling back")
        return None
    
    logging.info(f"Rule-based match: index {best_idx}, score {best_score}")
    return best_idx

def rerank_fusion(top_results, rule_idx, data):
    """
    Chọn FAQ tốt nhất từ semantic search và rule-based
    Input: top_results (list of (idx, score)), rule_idx (int or None), data (list)
    Output: best index (int) hoặc None
    """
    if rule_idx is not None:
        logging.info(f"Selected index {rule_idx} from rule-based matching")
        return rule_idx
    
    # Nếu rule-based không tìm thấy, lấy top-1 từ semantic search
    if top_results:
        best_idx = top_results[0][0]
        logging.info(f"Selected index {best_idx} from semantic search")
        return best_idx
    
    logging.warning("No match after reranking")
    return None

def postprocess_answer(answer, entities):
    """
    Chèn thực thể vào câu trả lời và định dạng
    Input: answer (str), entities (list of (entity, type))
    Output: formatted answer (str)
    """
    for entity, entity_type in entities:
        answer = answer.replace(f'[{entity_type}]', entity)
    
    logging.info(f"Formatted answer: {answer}")
    return answer

def detect_irrelevant_questions(query):
   
    query_lower = query.lower().strip()
    
    # Các mẫu câu hỏi không liên quan
    irrelevant_patterns = [
        # Chào hỏi cá nhân
        r'chào\s+(bạn|anh|chị|em|cô|chú|bác)',
        r'xin\s+chào',
        r'hello|hi|hey',
        r'chào\s+bot',
        
        # Hỏi sức khỏe, cảm xúc
        r'bạn\s+(khỏe|sao|thế\s+nào)',
        r'bot\s+(khỏe|sao|thế\s+nào)',
        r'bạn\s+có\s+khỏe\s+không',
        r'bạn\s+thế\s+nào',
        
        # Hỏi sở thích, cá nhân
        r'bạn\s+thích\s+gì',
        r'bạn\s+có\s+thích',
        r'sở\s+thích\s+của\s+bạn',
        r'bạn\s+thích\s+ăn\s+gì',
        r'bạn\s+thích\s+chơi\s+gì',
        
        # Hỏi thời tiết, tin tức
        r'thời\s+tiết\s+hôm\s+nay',
        r'tin\s+tức',
        r'báo\s+chí',
        r'tivi',
        
        # Hỏi về AI, bot
        r'bạn\s+là\s+ai',
        r'bạn\s+là\s+gì',
        r'bạn\s+có\s+phải\s+ai',
        r'bot\s+là\s+gì',
        
        # Hỏi linh tinh
        r'kể\s+chuyện',
        r'hát\s+bài',
        r'làm\s+thơ',
        r'đố\s+vui',
        r'chơi\s+game'
    ]
    
    import re
    for pattern in irrelevant_patterns:
        if re.search(pattern, query_lower):
            return True
    
    return False

def get_irrelevant_response(query):
    """
    Trả về câu trả lời cho các câu hỏi không liên quan
    Input: query (str)
    Output: response (str)
    """
    query_lower = query.lower().strip()
    
    # Chào hỏi
    if any(word in query_lower for word in ['chào', 'hello', 'hi', 'xin chào']):
        return "👋 Xin chào! Tôi là ChatBot hỗ trợ nội bộ của công ty. Tôi có thể giúp bạn trả lời các câu hỏi về quy định, chính sách công ty. Bạn có câu hỏi gì về công việc không?"
    
    # Hỏi sức khỏe
    elif any(word in query_lower for word in ['khỏe', 'sao', 'thế nào']):
        return "😊 Cảm ơn bạn đã quan tâm! Tôi là AI nên không có cảm xúc như con người. Tôi luôn sẵn sàng hỗ trợ bạn với các câu hỏi về quy định công ty. Bạn cần tìm hiểu gì về chính sách công ty không?"
    
    # Hỏi sở thích
    elif any(word in query_lower for word in ['thích', 'sở thích']):
        return "🤖 Tôi là AI nên không có sở thích cá nhân. Sở thích của tôi là giúp đỡ nhân viên tìm hiểu về quy định và chính sách công ty. Bạn có câu hỏi gì về công việc không?"
    
    # Hỏi về AI/bot
    elif any(word in query_lower for word in ['là ai', 'là gì', 'ai', 'bot']):
        return "🤖 Tôi là ChatBot hỗ trợ nội bộ, được thiết kế để trả lời các câu hỏi về quy định, chính sách công ty. Tôi có thể giúp bạn tìm hiểu về giờ làm việc, đồng phục, đóng gói hàng hóa, v.v. Bạn cần hỗ trợ gì?"
    
    # Câu hỏi linh tinh khác
    else:
        return "😅 Tôi chỉ có thể hỗ trợ các câu hỏi liên quan đến quy định và chính sách công ty. Bạn có thể hỏi tôi về:\n• Giờ làm việc\n• Chính sách đồng phục\n• Quy định đóng gói hàng hóa\n• Chính sách nghỉ phép\n• Và nhiều chủ đề khác..."

def detect_typos_and_suggest(query):
    """
    Phát hiện lỗi chính tả và đưa ra gợi ý
    Input: query (str)
    Output: suggestion (str) nếu có lỗi chính tả, None nếu không
    """
    query_lower = query.lower().strip()
    
    # Từ khóa chính xác và các biến thể có thể sai chính tả
    corrections = {
        # Giờ làm việc
        'giờ làm việc': ['giờ làm việc', 'thời gian làm việc', 'giờ đi làm'],
        'đồng phục': ['đồng phục', 'quần áo công ty', 'trang phục'],
        'đóng gói': ['đóng gói', 'bao bì', 'đóng gói hàng hóa'],
        'nghỉ phép': ['nghỉ phép', 'nghỉ việc', 'nghỉ có lương'],
        'lương': ['lương', 'tiền lương', 'thù lao'],
        'bảo hiểm': ['bảo hiểm', 'bảo hiểm xã hội', 'bảo hiểm y tế'],
        'hàng hóa': ['hàng hóa', 'sản phẩm', 'hàng gửi'],
        'vận chuyển': ['vận chuyển', 'gửi hàng', 'chuyển phát'],
        'internet': ['internet', 'mạng', 'wifi'],
        'đào tạo': ['đào tạo', 'học tập', 'training'],
        'khách hàng': ['khách hàng', 'client', 'customer']
    }
    
    # Tìm từ khóa gần đúng
    for correct_word, variants in corrections.items():
        for variant in variants:
            if variant in query_lower and correct_word not in query_lower:
                return f"💡 Có thể bạn muốn hỏi về **{correct_word}**? Tôi có thể giúp bạn tìm hiểu về chủ đề này."
    
    return None

def handle_no_match():
    """Trả về thông báo khi không tìm thấy kết quả"""
    answer = "❌ Không tìm thấy thông tin phù hợp với câu hỏi của bạn.\n\n💡 Bạn có thể thử:\n• Diễn đạt lại câu hỏi một cách rõ ràng hơn\n• Sử dụng từ khóa chính xác\n• Hỏi về các chủ đề như: giờ làm việc, đồng phục, đóng gói hàng hóa, nghỉ phép, lương, bảo hiểm\n\n🤖 Tôi luôn sẵn sàng hỗ trợ bạn!"
    logging.info(f"No-match answer: {answer}")
    return answer

def check_topic_query(query, json_files):
    """
    Kiểm tra xem query có phải là hỏi về chủ đề (title) không
    Input: query (str), json_files (list of file paths)
    Output: answer (str) nếu tìm thấy chủ đề, None nếu không
    """
    query_lower = query.lower().strip()
    logging.info(f"Checking topic query: {query}")
    
    # Các từ khóa để nhận diện câu hỏi về chủ đề
    topic_keywords = [
        'gồm những gì', 'có gì', 'bao gồm', 'liệt kê', 'kể tên', 'nêu ra',
        'các loại', 'danh sách', 'chi tiết', 'thông tin', 'quy định',
         'hướng dẫn', 'cách thức', 'quy trình', 'tất cả về', 'mọi thứ'
         'mọi thứ về'
    ]
    
    # Kiểm tra xem có phải câu hỏi về chủ đề không
    is_topic_query = any(keyword in query_lower for keyword in topic_keywords)
    
    # Nếu không có từ khóa chủ đề, kiểm tra xem có phải câu hỏi tổng quát không
    if not is_topic_query:
        general_questions = [
            'có những gì', 'gồm gì', 'bao gồm gì', 'có cái gì',
            'thông tin gì', 'quy định gì', 'chính sách gì','nêu','nêu ra'
            ,'tất tần tật ','mọi thứ'
        ]
        is_topic_query = any(q in query_lower for q in general_questions)
    
    if not is_topic_query:
        return None
    
    # Tìm kiếm chủ đề phù hợp
    best_match = None
    best_similarity = 0
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data['data']:
                    title = item['title'].lower()
                    
                    # Kiểm tra similarity giữa query và title
                    similarity = SequenceMatcher(None, query_lower, title).ratio()
                    
                    # Kiểm tra từ khóa chung
                    keyword_match = 0
                    query_words = query_lower.split()
                    title_words = title.split()
                    
                    for q_word in query_words:
                        for t_word in title_words:
                            if q_word in t_word or t_word in q_word:
                                keyword_match += 1
                                break
                    
                    # Tính combined score
                    combined_score = similarity + 0.1 * keyword_match
                    
                    # Nếu similarity cao hoặc có từ khóa chính trong title
                    if combined_score > best_similarity and (combined_score > 0.5 or keyword_match > 0):
                        best_similarity = combined_score
                        best_match = {
                            'title': item['title'],
                            'contexts': [paragraph['context'] for paragraph in item['paragraphs']],
                            'file': json_file,
                            'score': combined_score
                        }
        except Exception as e:
            logging.error(f"Error reading {json_file}: {e}")
    
    if best_match and best_similarity > 0.4:
        # Tạo câu trả lời tổng hợp
        answer = f"📋 **{best_match['title']}**\n\n"
        for i, context in enumerate(best_match['contexts'], 1):
            answer += f"**{i}.** {context}\n\n"
        
        logging.info(f"Topic match found: {best_match['title']} (score: {best_similarity:.3f})")
        return answer
    
    return None

def qa_pipeline(query, json_files):
    """
    Pipeline xử lý query và trả về câu trả lời
    Input: query (str), json_files (list of file paths)
    Output: answer (str)
    """
    # Bước 1: Kiểm tra câu hỏi không liên quan
    if detect_irrelevant_questions(query):
        logging.info(f"Irrelevant question detected: {query}")
        return get_irrelevant_response(query)
    
    # Bước 2: Kiểm tra lỗi chính tả và đưa ra gợi ý
    typo_suggestion = detect_typos_and_suggest(query)
    if typo_suggestion:
        logging.info(f"Typo suggestion provided for: {query}")
        return typo_suggestion
    
    # Bước 3: Kiểm tra câu hỏi về chủ đề
    topic_answer = check_topic_query(query, json_files)
    if topic_answer:
        return topic_answer
    
    # Bước 4: Exact question check
    answer = exact_question_check(query, json_files)
    if answer:
        return answer
    
    # Bước 5: Semantic search với fallback
    data, vectorizer, doc_vectors = load_data(json_files)
    if not data or vectorizer is None:
        return handle_no_match()
    
    # Preprocessing
    processed_query = preprocess_query(query)
    
    # NER extraction
    entities = ner_extraction(query)
    
    # Semantic search
    top_results = semantic_search(query, data, vectorizer, doc_vectors, entities)
    
    # Rule-based matching
    best_idx = rule_based_matching(query, data, top_results, entities)
    
    # Reranking / Fusion
    final_idx = rerank_fusion(top_results, best_idx, data)
    
    # Postprocessing với fallback thông minh
    if final_idx is None:
        # Kiểm tra xem có phải câu hỏi quá ngắn hoặc không rõ ràng không
        if len(processed_query.split()) < 2:
            return "🤔 Câu hỏi của bạn hơi ngắn. Bạn có thể mô tả chi tiết hơn về vấn đề cần hỗ trợ không?\n\n💡 Ví dụ: 'Giờ làm việc của công ty như thế nào?' thay vì chỉ 'giờ làm việc'"
        
        # Kiểm tra xem có phải câu hỏi về chủ đề không rõ ràng không
        if any(word in processed_query for word in ['gì', 'nào', 'sao', 'thế nào']):
            return "🤔 Bạn có thể cụ thể hơn về chủ đề cần tìm hiểu không?\n\n💡 Tôi có thể giúp bạn về:\n• Giờ làm việc\n• Chính sách đồng phục\n• Quy định đóng gói hàng hóa\n• Chính sách nghỉ phép\n• Và nhiều chủ đề khác..."
        
        return handle_no_match()
    
    answer = data[final_idx]['answer']
    return postprocess_answer(answer, entities)

# Flask Web App
from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

# Initialize data
json_files = [
    r"D:\du_an_chatBot_noi_bo\data\duLieuCongTy.json",
    r"D:\du_an_chatBot_noi_bo\data\data_chuDe_1.json",
    r"D:\du_an_chatBot_noi_bo\data\data_chuDe_2.json",
    r"D:\du_an_chatBot_noi_bo\data\data_chuDe_3.json",
    r"D:\du_an_chatBot_noi_bo\data\data_chuDe_4.json",
    r"D:\du_an_chatBot_noi_bo\data\data_chuDe_5.json",
    r"D:\du_an_chatBot_noi_bo\data\data_chuDe_7.json",
    r"D:\du_an_chatBot_noi_bo\data\data_chuDe_8.json",
    r"D:\du_an_chatBot_noi_bo\data\data_chuDe_9.json"
]

@app.route('/')
def index():
    """Trang chủ"""
    return render_template('index.html')

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """API endpoint để hỏi câu hỏi"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Vui lòng nhập câu hỏi'}), 400
        
        # Sử dụng logic có sẵn trong qa_pipeline
        answer = qa_pipeline(question, json_files)
        
        return jsonify({
            'question': question,
            'answer': answer,
            'status': 'success'
        })
    
    except Exception as e:
        logging.error(f"Error processing question: {e}")
        return jsonify({'error': 'Có lỗi xảy ra khi xử lý câu hỏi'}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'API đang hoạt động bình thường'})

# Example usage (chỉ chạy khi không phải Flask app)
if __name__ == "__main__":
    # import sys
    
    # # Kiểm tra nếu có argument 'web' thì chạy Flask app
    # if len(sys.argv) > 1 and sys.argv[1] == 'web':
        print("🚀 Starting Flask web app...")
        print("📱 Open your browser and go to: http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    # else:
    #     # Chạy test như cũ
    #     query = "Cách giữ đồng phục luôn sạch sẽ   "
    #     answer = qa_pipeline(query, json_files)
        
    #     # Safe printing with encoding handling
    #     os.system('chcp 65001 >nul 2>&1')  # Set console to UTF-8
        
    #     try:
    #         print(f"Query: {query}")
    #         print(f"Answer: {answer}")
    #     except:
    #         # Fallback: write to file
    #         with open('result.txt', 'w', encoding='utf-8') as f:
    #             f.write(f"Query: {query}\n")
    #             f.write(f"Answer: {answer}\n")
    #         print("Results saved to result.txt")