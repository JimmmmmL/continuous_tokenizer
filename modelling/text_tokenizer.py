from modelling.bert_mae import TextMAE, TextMAEArgs

def TextMAE_512(**kwargs):
    return TextMAE(TextMAEArgs(**kwargs))


VQ_models = {
    'TextMAE-512': TextMAE_512,  # for text tokenizer
}