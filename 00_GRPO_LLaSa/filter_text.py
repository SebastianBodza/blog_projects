import re 
european_pattern = r'^[Ééèàíáîśûúôśïçšâëνã\ta-zA-ZäöüßÄÖÜ\s\d\-_.:,;#\'+*?`´=)(/&%$§"!°^<>\W]*$'

def filter_text(example):
    def count_max_repetition(text):
        cleaned_text = re.sub(r'[,.]', '', text)
        words = cleaned_text.split()
        
        if not words: return 0
            
        max_repetition = 1
        current_repetition = 1
        current_word = words[0]
        
        for word in words[1:]:
            if word == current_word:
                current_repetition += 1
                max_repetition = max(max_repetition, current_repetition)
            else:
                current_repetition = 1
                current_word = word
                
        return max_repetition

    def has_excessive_char_repetition(text, default_threshold=5, underscore_threshold=9):
        words = text.split()
        
        for word in words:
            if len(word) < 2:
                continue
                
            current_char = word[0]
            current_repetition = 1
            for char in word[1:]:
                if char == current_char:
                    current_repetition += 1
                    if current_char == '_' and current_repetition > underscore_threshold:
                        return True
                    elif current_repetition > default_threshold:
                        return True
                else:
                    current_repetition = 1
                    current_char = char
                    
        return False


    def has_non_german_chars(text):
        return not bool(re.match(european_pattern, text))

    def has_chinese_chars(text):
        chinese_pattern = (
            r'[\u4e00-\u9fff]'
        )
        return bool(re.search(chinese_pattern, text))
    
    def calculate_unique_word_percentage(text):
        words = text.split()
        if not words:
            return 0
        return (len(set(words)) / len(words)) * 100

    def process_single_text(text):
        if count_max_repetition(text) >= 5:
            return False
        if has_excessive_char_repetition(text):
            return False
        if has_chinese_chars(text):
            return False
        if has_non_german_chars(text):
            return False
        if calculate_unique_word_percentage(text) <= 60:
            return False
        return True

    if isinstance(example['text'], list):
        return [process_single_text(text) for text in example['text']]
    return process_single_text(example['text'])