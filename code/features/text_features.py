"""Deterministic question/candidate features; no labels, IDs, or external models."""
import ast
import math
import re
from collections import Counter
from fractions import Fraction

import numpy as np
from common import parse_answer

NUMBER = re.compile(r'[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?')
WORDS = re.compile(r'[a-z]+')
EQUATION = re.compile(r'(?<![\w.])([-+\d.][\d.\s+*/()\-]{0,75})\s*=\s*([-+]?\d+(?:\.\d+)?(?:/\d+)?)')
STOP = set('the a an is are was were to of and in for it we this that by be with as on how what'.split())
TERMS = ('therefore','however','instead','actually','mistake','incorrect','correct','wait','assume',
         'approximately','about','round','remaining','left','total','each','per','percent','discount',
         'original','final','answer','step','verify','check','not','must','could','if','so')


def _numbers(text):
    result = []
    for token in NUMBER.findall(text):
        try:
            value = float(Fraction(token.replace(',', '')))
            if math.isfinite(value) and abs(value) < 1e15:
                result.append(value)
        except (ValueError, ZeroDivisionError, OverflowError):
            pass
    return result


def _numeric_answer(text):
    parsed = parse_answer(text)
    try:
        return float(Fraction(parsed)) if parsed is not None else None
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def _arith_value(node, depth=0):
    if depth > 12:
        raise ValueError('Expression too deep')
    if isinstance(node, ast.Expression):
        return _arith_value(node.body, depth+1)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _arith_value(node.operand, depth+1)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left, right = _arith_value(node.left, depth+1), _arith_value(node.right, depth+1)
        if isinstance(node.op, ast.Add): return left+right
        if isinstance(node.op, ast.Sub): return left-right
        if isinstance(node.op, ast.Mult): return left*right
        return left/right
    raise ValueError('Unsupported expression')


def _equations(text):
    clean = text.replace(',', '').replace('\\times', '*').replace('×', '*').replace('\\div', '/').replace('÷', '/')
    clean = re.sub(r'\\(?:d?frac)\{([^{}]+)\}\{([^{}]+)\}', r'((\1)/(\2))', clean)
    clean = clean.replace('**', '').replace('$', '')
    checked = good = bad = 0
    last_good = 0.
    for expression, expected in EQUATION.findall(clean):
        if not any(op in expression for op in '+-*/'):
            continue
        try:
            actual = _arith_value(ast.parse(expression.strip(), mode='eval'))
            expected = float(Fraction(expected))
            if not math.isfinite(actual) or abs(actual) > 1e15: continue
            matched = abs(actual-expected) <= 1e-5 * max(1.,abs(expected))
            checked += 1
            good += matched
            bad += not matched
            last_good = float(matched)
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
            pass
    return [checked, good, bad, good/max(1,checked), bad/max(1,checked), last_good]


def extract_features(bank, return_metadata=False):
    """Return float32 [N,4,D]; only supplied question/candidate texts are used."""
    if not bank:
        dummy=[{'question':'', 'candidates':[{'text':''} for _ in range(4)]}]
        features,names,groups=extract_features(dummy,True)
        empty=features[:0]
        return (empty,names,groups) if return_metadata else empty
    rows, names, groups = [], [], []
    for row in bank:
        question = row['question'].lower()
        qnumbers = _numbers(question)
        qwords = set(WORDS.findall(question))-STOP
        candidates = row['candidates']
        answers = [_numeric_answer(c['text']) for c in candidates]
        answer_keys = [parse_answer(c['text']) for c in candidates]
        counts = Counter(a for a in answer_keys if a is not None)
        lengths = np.array([len(c['text']) for c in candidates], dtype=float)
        max_vote = max(counts.values(), default=0)
        unique_answers = len(counts)
        result = []
        for j, candidate in enumerate(candidates):
            text = candidate['text']
            lower = text.lower()
            words = WORDS.findall(lower)
            word_set = set(words)-STOP
            nums = _numbers(text)
            a = answers[j]
            valid = a is not None and math.isfinite(a)
            a = a if valid else 0.
            current_names, current_groups, values = [], [], []
            def add(group, **items):
                for name, value in items.items():
                    current_names.append(name); current_groups.append(group); values.append(float(value))
            add('style', log_chars=np.log1p(len(text)), log_words=np.log1p(len(words)),
                log_lines=np.log1p(text.count('\n')), log_digits=np.log1p(sum(x.isdigit() for x in text)),
                log_numbers=np.log1p(len(nums)), unique_numbers=len(set(nums))/max(1,len(nums)),
                unique_words=len(set(words))/max(1,len(words)),
                boxed=float('\\boxed{' in text), hash_final=float('####' in text),
                hash_count=text.count('####'), placeholder=float('<numeric answer>' in lower),
                bold_count=np.log1p(text.count('**')), heading_count=np.log1p(text.count('###')),
                dollar_count=np.log1p(text.count('$')), equals_count=np.log1p(text.count('=')),
                frac_count=np.log1p(text.count('\\frac')), question_count=text.count('?'),
                exclamation_count=text.count('!'), parentheses_count=text.count('('),
                colon_count=np.log1p(text.count(':')), minus_count=np.log1p(text.count('-')),
                last_box=float('\\boxed' in text[-120:]), last_hash=float('####' in text[-120:]))
            for term in TERMS:
                add('style', **{f'word_{term}':float(bool(re.search(r'\b'+term+r'\b',lower)))})
            add('question', question_length=np.log1p(len(question)), question_numbers=len(qnumbers),
                question_overlap=len(qwords & word_set)/max(1,len(qwords)),
                unexplained_words=len(word_set-qwords)/max(1,len(word_set)),
                question_percent=float('%' in question or 'percent' in question),
                question_money=float('$' in question or 'dollar' in question),
                question_time=float(any(w in qwords for w in ['hour','hours','minutes','days','weeks'])),
                question_remaining=float(any(w in qwords for w in ['left','remaining','remain'])))
            add('answer', valid=float(valid), negative=float(a<0), integer=float(a==round(a)),
                zero=float(a==0), log_answer=np.log1p(abs(a)), signed_log_answer=np.sign(a)*np.log1p(abs(a)),
                answer_in_question=float(any(abs(a-v)<1e-7 for v in qnumbers)),
                answer_is_last_number=float(bool(nums) and abs(a-nums[-1])<1e-7),
                answer_repeated=sum(abs(a-v)<1e-7 for v in nums),
                q_sum_match=float(bool(qnumbers) and abs(a-sum(qnumbers))<1e-7),
                q_min_ratio=a/max(1.,abs(min(qnumbers,default=0.))),
                q_max_ratio=a/max(1.,abs(max(qnumbers,default=0.))))
            pair_matches = [0,0,0,0]
            for x in qnumbers[:12]:
                for y in qnumbers[:12]:
                    expressions = (x+y, x-y, x*y, x/y if y else float('inf'))
                    for k,value in enumerate(expressions):
                        pair_matches[k] += abs(a-value) < 1e-6*max(1,abs(a))
            for name,value in zip(('sum','difference','product','quotient'), pair_matches):
                add('answer',**{f'question_pair_{name}':value})
            equation_values = _equations(text)
            for name,value in zip(('eq_checked','eq_good','eq_bad','eq_good_fraction','eq_bad_fraction','eq_last_good'),equation_values):
                add('arithmetic',**{name:value})
            vote = counts.get(answer_keys[j],0) if valid else 0
            other_answers = [v for k,v in enumerate(answers) if k!=j and v is not None and math.isfinite(v)]
            add('consensus', votes=vote, is_majority=float(vote==max_vote), max_votes=max_vote,
                unique_answers=unique_answers, valid_count=sum(v is not None for v in answers),
                other_answer_log_distance=np.mean([np.log1p(abs(a-v)) for v in other_answers]) if other_answers else 0,
                answer_larger_count=sum(a>v for v in other_answers), answer_smaller_count=sum(a<v for v in other_answers),
                answer_near_other=sum(abs(a-v)<=max(1,abs(a))*.01 for v in other_answers),
                length_relative=(len(text)-lengths.mean())/max(1.,lengths.std()),
                length_rank=sum(len(text)>v for v in lengths),
                answer_group_mean_length=float(np.mean([lengths[k] for k in range(4) if answer_keys[k]==answer_keys[j]]))/max(1.,lengths.mean()),
                consensus_loglength=vote*np.log1p(len(text)))
            names, groups = current_names,current_groups
            result.append(values)
        rows.append(result)
    raw = np.asarray(rows,dtype=np.float32)
    raw=np.nan_to_num(raw,nan=0.,posinf=1e6,neginf=-1e6)
    raw=np.clip(raw,-1e6,1e6)
    # Candidate-relative features remove question-specific scale and reveal disagreements.
    relative=raw-raw.mean(axis=1,keepdims=True) if len(bank) else raw.copy()
    features=np.concatenate([raw,relative],axis=-1)
    if return_metadata:
        return features,names+[n+'_relative' for n in names],groups+groups
    return features
