"""Development observations for text models; never accesses test references."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from common import parse_answer
from training_utils import load_supervised
from methods.text import candidate_scores
from features.text_features import extract_features

bank,arrays,targets=load_supervised('validation')
hard=np.asarray([len({parse_answer(c['text']) for c in row['candidates'] if parse_answer(c['text']) is not None})>=2 for row in bank])
results=[]
for path in sorted(Path('runs/text').glob('*/model.npz')):
    scores=candidate_scores(bank,arrays,path)
    choices=scores.argmax(axis=1)
    correctness=targets[np.arange(len(bank)),choices].astype(bool)
    valid=scores>-1e5
    record={'checkpoint':str(path),'correct':int(correctness.sum()),'n':len(bank),
            'accuracy':float(correctness.mean()*100),'hard_correct':int(correctness[hard].sum()),
            'hard_n':int(hard.sum()),'hard_accuracy':float(correctness[hard].mean()*100),
            'valid_score_percentiles':np.percentile(scores[valid],[0,5,50,95,100]).tolist(),
            'asset_bytes':path.stat().st_size}
    results.append(record)
Path('runs/text/validation_observations.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf8')
print(json.dumps(results,indent=2))

recommended=Path('runs/text/no_arithmetic/model.npz')
if recommended.exists():
    scores=candidate_scores(bank,arrays,recommended)
    chosen=scores.argmax(axis=1)
    failures=[]
    for i in np.flatnonzero(targets[np.arange(len(bank)),chosen]==0)[:5]:
        row=bank[i]
        parsed=[parse_answer(c['text']) for c in row['candidates']]
        votes=[parsed.count(a) if a is not None else 0 for a in parsed]
        gold=parsed[int(np.flatnonzero(targets[i])[0])]
        failures.append({'id':row['id'],'question':row['question'],'selected':int(chosen[i]),
                         'correct_candidates':np.flatnonzero(targets[i]).tolist(),
                         'correct_final_answer':gold,'parsed_answers':parsed,'answer_votes':votes,
                         'scores':scores[i].tolist(),'candidate_texts':[c['text'] for c in row['candidates']],
                         'observation':'The selected final answer is wrong under the fixed numeric scorer. Full reasoning validity was not automatically assessed.'})
    Path('runs/text/validation_failure_examples.json').write_text(json.dumps(failures,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
