import MeCab

def tokenize_korean(sentence):
    tokenizer = MeCab.Tagger()
    # mecab이 반환하는 형태소 분석 결과에서 단어만 추출하여 리스트로 반환
    tokens = [token.split('\t')[0] for token in tokenizer.parse(sentence).splitlines()[:-1]]
    return tokens

sentence = "한국어 문장이 들어왔을 때, 문장을 형태소 단위로 나누는 python 코드를 짜 줘."
tokens = tokenize_korean(sentence)
print(tokens)
