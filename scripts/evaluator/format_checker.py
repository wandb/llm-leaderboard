import re

#mawps, mgsm
def is_all_digit(text):
    return text.isdigit()

#jmmlu, mmlu
def is_one_of_ABCD(text):
    return text in {'A', 'B', 'C', 'D'}

#JBLiMP
def is_a_b(text):
    return text in {'a', 'b'}

#jcommonsenseqa
def is_0_4(text):
    return text in {'0', '1', '2', '3', '4'}

#jcola, JCommonsenseMorality
def is_0_1(text):
    return text in {'0', '1'}

#janli
def is_entailment2_format(text):
    return text in {'entailment','non-entailment'}

#jnli, jsick, jamp
def is_entailment3_format(text):
    return text in {'entailment','contradiction','neutral'}

#jsem
def is_jsem_format(text):
    return text in {'yes','no','unknown','undef'}

#wiki_ner
def is_wiki_ner_format(text):
    allowed_tags = {'組織名', '人名', '地名', '固有物名', '日付表現', '時刻表現', '金額表現', '割合表現'}    
    pattern = re.compile(r'^(.+?)\（(' + '|'.join(allowed_tags) + r')\）$')
    segments = text.split()
    for segment in segments:
        if not pattern.match(segment):
            return False
    return True

#wiki_dependency
def is_wiki_dependecy_format(text):
    pattern = re.compile(r'^.+\s*->\s*.+$')
    lines = text.split('\n')
    for line in lines:
        if not pattern.match(line):
            return False
    return True

#chabsa
def is_chabsa_format(text):
    pattern = re.compile(r'(\w+)\s+(positive|neutral|negative)')
    
    lines = text.split('\n')
    for line in lines:
        if not pattern.match(line):
            return False
    return True

format_check_dict = {
    'mawps': is_all_digit,
    'mgsm': is_all_digit,
    'jmmlu': is_one_of_ABCD,
    'mmlu': is_one_of_ABCD,
    'JBLiMP': is_a_b,
    'jcommonsenseqa': is_0_4,
    'jcola': is_0_1,
    'JCommonsenseMorality': is_0_1,
    'janli': is_entailment2_format,
    'jnli': is_entailment3_format,
    'jsick': is_entailment3_format,
    'jamp': is_entailment3_format,
    'jsem': is_jsem_format,
    'wiki_ner': is_wiki_ner_format,
    'wiki_dependency': is_wiki_dependecy_format,
    'chabsa': is_chabsa_format
}