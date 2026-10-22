"""One reusable 832-token cache boundary; not a context qualification."""
import json
from pathlib import Path
import time
import urllib.request

root=Path(__file__).resolve().parents[1]
def metrics():
    with urllib.request.urlopen('http://127.0.0.1:8000/metrics',timeout=10) as r:
        return r.read().decode()
def count(text,name):
    return sum(float(line.split()[-1]) for line in text.splitlines() if line.startswith(name+'{'))
def chat():
    data={'model':'Intel/Qwen3.8-Flash-Next-W4A16-AutoRound',
          'messages':[{'role':'user','content':(' alpha'*1100)+'\nReply with just DONE.'}],
          'temperature':0,'max_tokens':8,'chat_template_kwargs':{'enable_thinking':False}}
    req=urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=180) as r:
        result=json.load(r)
    assert 'DONE' in result['choices'][0]['message']['content']
    return result
before=metrics()
a=chat()
mid=metrics()
b=chat()
time.sleep(1)
after=metrics()
hits_before=count(before,'vllm:prefix_cache_hits_total')
hits_after=count(after,'vllm:prefix_cache_hits_total')
record={'first':a,'second':b,'prefix_hit_delta':hits_after-hits_before}
(root/'evidence/apc-reuse.json').write_text(json.dumps(record,indent=2)+'\n')
(root/'evidence/apc-metrics-before.txt').write_text(before)
(root/'evidence/apc-metrics-after.txt').write_text(after)
assert a['usage']['prompt_tokens']>832
assert record['prefix_hit_delta']>=832,record
assert a['choices'][0]['message']['content']==b['choices'][0]['message']['content']
print(json.dumps({'passed':True,'prompt_tokens':a['usage']['prompt_tokens'],'prefix_hit_delta':record['prefix_hit_delta']}))
