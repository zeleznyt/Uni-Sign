# coding: utf-8
"""
This module holds various MT evaluation metrics.
"""

from external_metrics import Rouge, sacrebleu, mscoco_rouge
import numpy as np
import json
import os
import re
from datetime import datetime

WER_COST_DEL = 3
WER_COST_INS = 3
WER_COST_SUB = 4
 

def chrf(references, hypotheses):
    """
    Character F-score from sacrebleu
    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    return (
        sacrebleu.corpus_chrf(hypotheses=hypotheses, references=references).score * 100
    )


def bleu(references, hypotheses, level='word'):
    """
    Raw corpus BLEU from sacrebleu (without tokenization)
    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    if level=='char':
        #split word
        references = [' '.join(list(r)) for r in references]
        hypotheses = [' '.join(list(r)) for r in hypotheses]
    bleu_scores = sacrebleu.raw_corpus_bleu(
        sys_stream=hypotheses, ref_streams=[references]
    ).scores
    scores = {}
    for n in range(len(bleu_scores)):
        scores["bleu" + str(n + 1)] = bleu_scores[n]
    return scores


def token_accuracy(references, hypotheses, level="word"):
    """
    Compute the accuracy of hypothesis tokens: correct tokens / all tokens
    Tokens are correct if they appear in the same position in the reference.
    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :param level: segmentation level, either "word", "bpe", or "char"
    :return:
    """
    correct_tokens = 0
    all_tokens = 0
    split_char = " " if level in ["word", "bpe"] else ""
    assert len(hypotheses) == len(references)
    for hyp, ref in zip(hypotheses, references):
        all_tokens += len(hyp)
        for h_i, r_i in zip(hyp.split(split_char), ref.split(split_char)):
            # min(len(h), len(r)) tokens considered
            if h_i == r_i:
                correct_tokens += 1
    return (correct_tokens / all_tokens) * 100 if all_tokens > 0 else 0.0


def sequence_accuracy(references, hypotheses):
    """
    Compute the accuracy of hypothesis tokens: correct tokens / all tokens
    Tokens are correct if they appear in the same position in the reference.
    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    assert len(hypotheses) == len(references)
    correct_sequences = sum(
        [1 for (hyp, ref) in zip(hypotheses, references) if hyp == ref]
    )
    return (correct_sequences / len(hypotheses)) * 100 if hypotheses else 0.0


# implement sltunet rouge eval  
def rouge_deprecated(references, hypotheses, level='word'):
    #beta 1.2
    rouge_score = 0
    n_seq = len(hypotheses)
    if level=='char':
        #split word
        references = [' '.join(list(r)) for r in references]
        hypotheses = [' '.join(list(r)) for r in hypotheses]
        
    for h, r in zip(hypotheses, references):
        rouge_score += mscoco_rouge.calc_score(hypotheses=[h], references=[r]) / n_seq

    return rouge_score * 100

def rouge(references, hypotheses, level='word'):
    if level=='char':
        hyp = [list(x) for x in hypotheses]
        ref = [list(x) for x in references]
    else:
        hyp = [x.split() for x in hypotheses]
        ref = [x.split() for x in references]
    a = Rouge.rouge([' '.join(x) for x in hyp], [' '.join(x) for x in ref])
    return a['rouge_l/f_score']*100

def wer_list(references, hypotheses):
    total_error = total_del = total_ins = total_sub = total_ref_len = 0

    for r, h in zip(references, hypotheses):
        res = wer_single(r=r, h=h)
        total_error += res["num_err"]
        total_del += res["num_del"]
        total_ins += res["num_ins"]
        total_sub += res["num_sub"]
        total_ref_len += res["num_ref"]

    wer = (total_error / total_ref_len) * 100
    del_rate = (total_del / total_ref_len) * 100
    ins_rate = (total_ins / total_ref_len) * 100
    sub_rate = (total_sub / total_ref_len) * 100

    return {
        "wer": wer,
        "del_rate": del_rate,
        "ins_rate": ins_rate,
        "sub_rate": sub_rate,
        # "del":total_del,
        # "ins":total_ins,
        # "sub":total_sub,
        # "ref_len":total_ref_len,
        # "error":total_error,
    }


def wer_single(r, h):
    r = r.strip().split()
    h = h.strip().split()
    edit_distance_matrix = edit_distance(r=r, h=h)
    alignment, alignment_out = get_alignment(r=r, h=h, d=edit_distance_matrix)

    num_cor = np.sum([s == "C" for s in alignment])
    num_del = np.sum([s == "D" for s in alignment])
    num_ins = np.sum([s == "I" for s in alignment])
    num_sub = np.sum([s == "S" for s in alignment])
    num_err = num_del + num_ins + num_sub
    num_ref = len(r)

    return {
        "alignment": alignment,
        "alignment_out": alignment_out,
        "num_cor": num_cor,
        "num_del": num_del,
        "num_ins": num_ins,
        "num_sub": num_sub,
        "num_err": num_err,
        "num_ref": num_ref,
    }


def edit_distance(r, h):
    """
    Original Code from https://github.com/zszyellow/WER-in-python/blob/master/wer.py
    This function is to calculate the edit distance of reference sentence and the hypothesis sentence.
    Main algorithm used is dynamic programming.
    Attributes:
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
    """
    d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8).reshape(
        (len(r) + 1, len(h) + 1)
    )
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                # d[0][j] = j
                d[0][j] = j * WER_COST_INS
            elif j == 0:
                d[i][0] = i * WER_COST_DEL
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitute = d[i - 1][j - 1] + WER_COST_SUB
                insert = d[i][j - 1] + WER_COST_INS
                delete = d[i - 1][j] + WER_COST_DEL
                d[i][j] = min(substitute, insert, delete)
    return d


def get_alignment(r, h, d):
    """
    Original Code from https://github.com/zszyellow/WER-in-python/blob/master/wer.py
    This function is to get the list of steps in the process of dynamic programming.
    Attributes:
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
        d -> the matrix built when calculating the editing distance of h and r.
    """
    x = len(r)
    y = len(h)
    max_len = 3 * (x + y)

    alignlist = []
    align_ref = ""
    align_hyp = ""
    alignment = ""

    while True:
        if (x <= 0 and y <= 0) or (len(alignlist) > max_len):
            break
        elif x >= 1 and y >= 1 and d[x][y] == d[x - 1][y - 1] and r[x - 1] == h[y - 1]:
            align_hyp = " " + h[y - 1] + align_hyp
            align_ref = " " + r[x - 1] + align_ref
            alignment = " " * (len(r[x - 1]) + 1) + alignment
            alignlist.append("C")
            x = max(x - 1, 0)
            y = max(y - 1, 0)
        elif x >= 1 and y >= 1 and d[x][y] == d[x - 1][y - 1] + WER_COST_SUB:
            ml = max(len(h[y - 1]), len(r[x - 1]))
            align_hyp = " " + h[y - 1].ljust(ml) + align_hyp
            align_ref = " " + r[x - 1].ljust(ml) + align_ref
            alignment = " " + "S" + " " * (ml - 1) + alignment
            alignlist.append("S")
            x = max(x - 1, 0)
            y = max(y - 1, 0)
        elif y >= 1 and d[x][y] == d[x][y - 1] + WER_COST_INS:
            align_hyp = " " + h[y - 1] + align_hyp
            align_ref = " " + "*" * len(h[y - 1]) + align_ref
            alignment = " " + "I" + " " * (len(h[y - 1]) - 1) + alignment
            alignlist.append("I")
            x = max(x, 0)
            y = max(y - 1, 0)
        else:
            align_hyp = " " + "*" * len(r[x - 1]) + align_hyp
            align_ref = " " + r[x - 1] + align_ref
            alignment = " " + "D" + " " * (len(r[x - 1]) - 1) + alignment
            alignlist.append("D")
            x = max(x - 1, 0)
            y = max(y, 0)

    align_ref = align_ref[1:]
    align_hyp = align_hyp[1:]
    alignment = alignment[1:]

    return (
        alignlist[::-1],
        {"align_ref": align_ref, "align_hyp": align_hyp, "alignment": alignment},
    )
    
def sableu(references, hypotheses, tokenizer):
    """
    Sacrebleu (with tokenization)

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    bleu_scores = sacrebleu.corpus_bleu(
        sys_stream=hypotheses, ref_streams=[references], tokenize=tokenizer,
    ).scores
    scores = {}
    for n in range(len(bleu_scores)):
        scores["bleu" + str(n + 1)] = bleu_scores[n]
        
    return scores

def standardized_bleu(references, hypotheses, effective_order=False):
    """
    SacreBLEU implementation aligned with the T5_for_SLT project.
    BLEU-1..4 are computed as BLEU scores with max_ngram_order=1..4, not
    as the individual n-gram precisions from a single BLEU-4 calculation.
    """
    from sacrebleu.metrics import BLEU

    print(
        "Standardized BLEU: "
        "tokenize=13a, smooth_method=exp, lowercase=False, "
        f"effective_order={effective_order}, max_ngram_order=1..4"
    )

    scores = {}
    for ngram_order in range(1, 5):
        bleu_score = BLEU(
            tokenize="13a",
            smooth_method="exp",
            effective_order=effective_order,
            lowercase=False,
            max_ngram_order=ngram_order,
        ).corpus_score(hypotheses, [references])
        scores["bleu" + str(ngram_order)] = bleu_score.score
    return scores


def standardized_bleu_report(references, hypotheses, effective_order=False):
    """
    SacreBLEU report with BLEU-1..4 and BLEU-4 precisions.
    references can be either one reference per sample:
        [ref_1, ref_2, ...]
    or multiple references per sample:
        [[ref_1a, ref_1b], [ref_2a], ...]
    Missing references are padded with the sample's last reference, matching
    sacrebleu's expected [num_refs x num_samples] corpus format.
    """
    from sacrebleu.metrics import BLEU

    if len(references) == 0:
        reference_streams = [[]]
    elif isinstance(references[0], (list, tuple)):
        ref_groups = [
            [ref.strip() if isinstance(ref, str) else "" for ref in refs]
            for refs in references
        ]
        ref_groups = [refs if len(refs) > 0 else [""] for refs in ref_groups]
        max_refs = max(len(refs) for refs in ref_groups)
        reference_streams = [
            [refs[ref_idx] if ref_idx < len(refs) else refs[-1] for refs in ref_groups]
            for ref_idx in range(max_refs)
        ]
    else:
        reference_streams = [[ref.strip() if isinstance(ref, str) else "" for ref in references]]

    norm_hyp = [hyp.strip() if isinstance(hyp, str) else "" for hyp in hypotheses]
    bleu_scores = {}
    bleu4 = None
    for ngram_order in range(1, 5):
        score = BLEU(
            tokenize="13a",
            smooth_method="exp",
            effective_order=effective_order,
            lowercase=False,
            max_ngram_order=ngram_order,
        ).corpus_score(norm_hyp, reference_streams)
        bleu_scores[f"bleu-{ngram_order}"] = score.score
        if ngram_order == 4:
            bleu4 = score

    for precision_idx, precision in enumerate(bleu4.precisions, start=1):
        bleu_scores[f"bleu-{precision_idx}_precision"] = precision

    return bleu_scores


def standardized_rouge_l(references, hypotheses):
    """
    Standardized ROUGE-L F1 using google-research rouge_score.
    Single-reference samples are scored directly. Multi-reference samples use
    max-over-references per prediction, then mean over samples.
    """
    from rouge_score import rouge_scorer

    print(
        "Standardized ROUGE: "
        "implementation=rouge_score, rouge_types=rougeL, use_stemmer=False, "
        "aggregation=mean_over_samples, multi_reference=max_over_references, scale=0..100"
    )

    def has_word_content(text):
        return isinstance(text, str) and bool(re.search(r"\w", text.strip(), flags=re.UNICODE))

    # Normalize input shape to one list of references per prediction.
    norm_hyp = [hyp.strip() if isinstance(hyp, str) else "" for hyp in hypotheses]
    if len(references) == 0:
        ref_groups = []
    elif isinstance(references[0], (list, tuple)):
        ref_groups = [[ref.strip() if isinstance(ref, str) else "" for ref in refs] for refs in references]
    else:
        ref_groups = [[ref.strip() if isinstance(ref, str) else ""] for ref in references]

    if len(norm_hyp) != len(ref_groups):
        raise ValueError(
            f"ROUGE expects the same number of hypotheses and references, "
            f"got {len(norm_hyp)} and {len(ref_groups)}."
        )

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    sample_scores = []
    for hyp, refs in zip(norm_hyp, ref_groups):
        # Punctuation-only / emoji-only strings can break or distort ROUGE; score them as zero.
        valid_refs = [ref for ref in refs if has_word_content(ref)]
        if not has_word_content(hyp) or len(valid_refs) == 0:
            sample_scores.append(0.0)
            continue
        # Multi-reference ROUGE uses the best matching reference for each prediction.
        sample_scores.append(
            max(scorer.score(target=ref, prediction=hyp)["rougeL"].fmeasure for ref in valid_refs)
        )

    return float(np.mean(sample_scores) * 100) if len(sample_scores) > 0 else 0.0


def translation_performance(
    txt_ref,
    txt_hyp,
    original_metric_implementation=False,
    bleu_effective_order=False,
):
    norm_hyp = [x.strip() if isinstance(x, str) else "" for x in txt_hyp]
    norm_ref = [x.strip() if isinstance(x, str) else "" for x in txt_ref]
    
    if original_metric_implementation:
        from rouge import Rouge as SLT_Rouge
        rouge = SLT_Rouge()

        # The rouge package can fail on strings that become token-empty after its own preprocessing
        # (e.g., punctuation/symbol-only). Keep this filtering only for legacy ROUGE.
        def _has_word_content(x):
            return isinstance(x, str) and bool(re.search(r"\w", x.strip(), flags=re.UNICODE))

        rouge_pairs = [(h, r) for h, r in zip(norm_hyp, norm_ref) if _has_word_content(h) and _has_word_content(r)]

        if len(rouge_pairs) > 0:
            rouge_hyp = [h for h, _ in rouge_pairs]
            rouge_ref = [r for _, r in rouge_pairs]
            scores = rouge.get_scores(rouge_hyp, rouge_ref, avg=True)
            rouge_l_f = scores['rouge-l']['f'] * 100
        else:
            rouge_l_f = 0.0

        tokenizer_args = '13a'
        # print('Signature: BLEU+case.mixed+numrefs.1+smooth.exp+tok.%s+version.1.4.2' % tokenizer_args)
        sableu_dict = sableu(references=norm_ref, hypotheses=norm_hyp, tokenizer=tokenizer_args)
    else:
        rouge_l_f = standardized_rouge_l(references=norm_ref, hypotheses=norm_hyp)
        sableu_dict = standardized_bleu(
            references=norm_ref,
            hypotheses=norm_hyp,
            effective_order=bleu_effective_order,
        )
    # print('BLEU', sableu_dict)
    # print('Signature: chrF2+case.mixed+numchars.6+numrefs.1+space.False+version.1.4.2')
    # print('Chrf', chrf(references=txt_ref, hypotheses=txt_hyp))
   
    print(sableu_dict)
    empty_hyp_count = sum(1 for h in norm_hyp if len(h) == 0)
    if empty_hyp_count > 0:
        print(f"Warning: {empty_hyp_count} empty hypotheses encountered during evaluation.")
    print(f"Rouge: {rouge_l_f:.2f}")
   
    # res = []
    # for n in range(4):
    #     res.append(f"{sableu_dict['bleu' + str(n + 1)]:.2f}")
    # res.append(f"{scores['rouge-l']['f']:.2f}")
    
    # print(" & ".join(res))

    return sableu_dict, float(rouge_l_f)

def islr_performance(txt_ref, txt_hyp):
    true_sample = 0
    for tgt_pre, tgt_ref in zip(txt_hyp, txt_ref):
        true_sample += (tgt_pre == tgt_ref)
    
    top1_acc_pi = true_sample / len(txt_hyp) * 100

    gt_dict = {}
    pred_dict = {}
    for tgt_pre, tgt_ref in zip(txt_hyp, txt_ref):
        if tgt_ref in gt_dict.keys():
            gt_dict[tgt_ref] += 1
            pred_dict[tgt_ref] += (tgt_pre == tgt_ref)
        else:
            gt_dict[tgt_ref] = 1
            pred_dict[tgt_ref] = (tgt_pre == tgt_ref)

    mean_acc_pc = []
    for key in gt_dict.keys():
        mean_acc_pc.append(pred_dict[key] / gt_dict[key])
    top1_acc_pc = np.array(mean_acc_pc).mean() * 100

    print(f"top1_acc_pi: {top1_acc_pi:.2f}")
    print(f"top1_acc_pc: {top1_acc_pc:.2f}")
    
    return top1_acc_pi, top1_acc_pc
   
    
   
