import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load('output/spm_unified_multilingual.model')

# Inspect
print(sp.get_piece_size())        # vocab size
print(sp.encode('English: The girl is reading. Amharic: ልጅቷ እያነበበች ነው። Oromo: Intaluma dubbisaa jirtti.', out_type=str))  # tokenize
print(sp.id_to_piece(5))         # see a specific token     