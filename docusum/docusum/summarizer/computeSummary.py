from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

def compute(sm) :
    model_name = 'google/pegasus-xsum'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)  # This line and the next slow it down
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
    try :
        batch = tokenizer.prepare_seq2seq_batch([sm], truncation=True, padding='longest').to(torch_device)
    except :
        return "This is nothing"
    translated = model.generate(**batch)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
                                
    return tgt_text[0]
