from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

def splitText(txt, s_len) : # This is needed in order to ensure the summarizer returns more than one sentence
    """ 
    Method that divides a summary into sections depending upon it length.
    These sections are then fed into the Pegasus Summarizer which provides
    summaries according to the sections length.
    """
    if s_len <= 1000 :
        return [txt]

    elif s_len <= 5000 :
        sen_list = []
        ind = 1000
        b_ind = 0
        while ind <= s_len :
            sen_list.append(txt[b_ind:ind])
            b_ind += 1000
            ind += 1000
        end = txt[b_ind : ]
        if len(end) >= 500 :
            sen_list.append(end)
        return sen_list

    elif s_len <= 10000 :
        sen_list = []
        ind = 2000
        b_ind = 0
        while ind <= s_len :
            sen_list.append(txt[b_ind:ind])
            b_ind += 2000
            ind += 2000
        end = txt[b_ind : ]
        if len(end) >= 500 :
            sen_list.append(end)
        return sen_list

    elif s_len <= 15000 :
        sen_list = []
        ind = 3000
        b_ind = 0
        while ind <= s_len :
            sen_list.append(txt[b_ind:ind])
            b_ind += 3000
            ind += 3000
        end = txt[b_ind : ]
        if len(end) >= 500 :
            sen_list.append(end)
        return sen_list

    elif s_len <= 20000 :
        sen_list = []
        ind = 4000
        b_ind = 0
        while ind <= s_len :
            sen_list.append(txt[b_ind:ind])
            b_ind += 4000
            ind += 4000
        end = txt[b_ind : ]
        if len(end) >= 500 :
            sen_list.append(end)
        return sen_list

    else :
        separation = int(s_len / 10)
        s_copy = separation
        sen_list = []
        while separation < s_len :
            sen_list.append(txt[separation-2000 : separation])
            separation += s_copy
        return sen_list


# Generate the summary
def compute(sm) :
    # Import the Pegasus Model
    model_name = 'google/pegasus-xsum'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
    sm_len = len(sm)

    sen_list = splitText(sm, sm_len) # Get sections to be summarized

    try :
        batches = []
        for s in sen_list : # Preparation
            batch = tokenizer.prepare_seq2seq_batch([s], truncation=True, padding='longest').to(torch_device)
            batches.append(batch)
    except :
        return ""

    temp = []
    for b in batches : # Summary generation
        translated = model.generate(**b)
        temp.append(translated)

    final_summary = []
    for t in temp : # Put together the summaries from the different sections
        final_summary.append(tokenizer.batch_decode(t, skip_special_tokens=True)[0])

    return final_summary



