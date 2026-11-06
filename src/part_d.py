from pathlib import Path
import json, re, time, sqlite3, sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'; POL=ROOT/'data/policies'; DB=ROOT/'data/instamart.db'; STATE=ROOT/'state'
OUT.mkdir(exist_ok=True); STATE.mkdir(exist_ok=True); sys.path.insert(0,str(ROOT)); import mock_llm

def chunk_docs(overlap=10):
    chunks=[]
    for path in sorted(POL.glob('*.txt')):
        words=path.read_text(encoding='utf-8').split(); start=0
        while start<len(words):
            end=min(start+40,len(words)); chunks.append({'source':path.name,'text':' '.join(words[start:end])})
            if end>=len(words): break
            start=end-overlap
    return chunks
class Retriever:
    def __init__(self,overlap=10):
        self.chunks=chunk_docs(overlap); self.vectorizer=TfidfVectorizer(); self.matrix=self.vectorizer.fit_transform([x['text'] for x in self.chunks])
    def retrieve(self,query,k=3):
        q=self.vectorizer.transform([query]); scores=cosine_similarity(q,self.matrix).ravel(); order=np.argsort(-scores,kind='stable')[:k]
        return [{'source':self.chunks[i]['source'],'score':float(scores[i]),'text':self.chunks[i]['text']} for i in order]
    def answer(self,question):
        hits=self.retrieve(question,3); context='\n'.join(f"[{x['source']}] {x['text']}" for x in hits)
        messages=[{'role':'system','content':'You are a POLICY ASSISTANT. Answer only from the supplied context. If the answer is not present, say: I could not find that information in the policy documents.'},{'role':'user','content':f'CONTEXT:\n{context}\nQUESTION:\n{question}'}]
        response=mock_llm.chat(messages); return response.choices[0].message.content,[x['source'] for x in hits]

def get_order(order_id):
    with sqlite3.connect(DB) as con:
        df=pd.read_sql_query('SELECT order_id,status,order_amount,delivery_minutes FROM orders WHERE order_id=?',con,params=(order_id,))
    if df.empty: return {'error':'order not found'}
    row=df.iloc[0].to_dict(); row['order_amount']=float(row['order_amount']); row['delivery_minutes']=None if pd.isna(row['delivery_minutes']) else int(row['delivery_minutes']); return row
def search_policy(query): return {'results':Retriever(10).retrieve(query,3)}
def calculate_refund(order_amount,delivery_minutes):
    amount=float(order_amount); mins=float(delivery_minutes) if delivery_minutes is not None else 0
    refund=round(amount,2) if mins>45 else 0.0
    return {'refund_inr':refund}
def _ledger(tool_name,order_id,refund_inr):
    record={'timestamp':datetime.now(timezone.utc).isoformat(),'tool':tool_name,'order_id':order_id,'refund_inr':round(float(refund_inr),2)}
    with (OUT/'refund_ledger.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(record)+'\n')
    return {'refund_inr':round(float(refund_inr),2)}
def issue_refund(order_id,refund_inr): return _ledger('issue_refund',order_id,refund_inr)
def escalate_to_human(order_id,refund_inr): return _ledger('escalate_to_human',order_id,refund_inr)
TOOLS_SPEC=[
 {'name':'get_order','description':'Look up one order in the trusted SQLite database.','parameters':{'type':'object','properties':{'order_id':{'type':'string'}},'required':['order_id']}},
 {'name':'search_policy','description':'Retrieve the most relevant policy chunks for a query.','parameters':{'type':'object','properties':{'query':{'type':'string'}},'required':['query']}},
 {'name':'calculate_refund','description':'Calculate a full refund only for delivery later than 45 minutes.','parameters':{'type':'object','properties':{'order_amount':{'type':'number'},'delivery_minutes':{'type':'number'}},'required':['order_amount','delivery_minutes']}},
 {'name':'issue_refund','description':'Append an approved automatic refund to the ledger.','parameters':{'type':'object','properties':{'order_id':{'type':'string'},'refund_inr':{'type':'number'}},'required':['order_id','refund_inr']}},
 {'name':'escalate_to_human','description':'Record a refund above INR 1000 for human approval.','parameters':{'type':'object','properties':{'order_id':{'type':'string'},'refund_inr':{'type':'number'}},'required':['order_id','refund_inr']}}]
TOOL_MAP={'get_order':get_order,'search_policy':search_policy,'calculate_refund':calculate_refund,'issue_refund':issue_refund,'escalate_to_human':escalate_to_human}
class SimulatedCrash(RuntimeError): pass
def _checkpoint(run_id,scenario_id,messages,tokens_used,step,tools_executed):
    payload={'run_id':run_id,'scenario_id':scenario_id,'messages':messages,'tokens_used':tokens_used,'step_number':step,'tools_executed':tools_executed}
    (STATE/f'{run_id}.json').write_text(json.dumps(payload,default=str,indent=2),encoding='utf-8')
def _continue(run_id,scenario_id,messages,tokens_used,step,tools_executed,max_iterations=6,token_limit=6000,time_limit_seconds=10,started=None,crash_after_tool_number=None):
    if started is None: started=time.monotonic()
    llm_calls=0; new_tools=0; steps=[]
    while llm_calls<max_iterations:
        if time.monotonic()-started>=time_limit_seconds:
            return {'status':'FALLBACK_TIMEOUT','final_answer':'FALLBACK_TIMEOUT: hand off to a human; time limit reached.','steps':steps,'llm_calls':llm_calls,'tokens_used':tokens_used,'tools_executed':tools_executed}
        response=mock_llm.chat(messages,tools=TOOLS_SPEC); llm_calls+=1; tokens_used+=response.usage.total_tokens
        if tokens_used>token_limit:
            return {'status':'FALLBACK_TOKEN_LIMIT','final_answer':'FALLBACK_TOKEN_LIMIT: hand off to a human; token budget exceeded.','steps':steps,'llm_calls':llm_calls,'tokens_used':tokens_used,'tools_executed':tools_executed}
        msg=response.choices[0].message
        if not msg.tool_calls:
            content=msg.content or 'FALLBACK: hand off to a human.'
            status=content.split(':',1)[0] if ':' in content else 'DONE'
            return {'status':status,'final_answer':content,'steps':steps,'llm_calls':llm_calls,'tokens_used':tokens_used,'tools_executed':tools_executed}
        assistant_call={'role':'assistant','content':None,'tool_calls':[{'id':tc.id,'name':tc.name,'arguments':tc.arguments} for tc in msg.tool_calls]}
        messages.append(assistant_call)
        for tc in msg.tool_calls:
            args=tc.arguments if isinstance(tc.arguments,dict) else json.loads(tc.arguments)
            tool=TOOL_MAP.get(tc.name)
            observation=tool(**args) if tool else {'error':'unknown tool'}
            tool_executed={'tool':tc.name,'arguments':args,'observation':observation}; steps.append(tool_executed); tools_executed.append(tc.name); new_tools+=1; step+=1
            messages.append({'role':'tool','tool_call_id':tc.id,'name':tc.name,'content':json.dumps(observation,default=str)})
            _checkpoint(run_id,scenario_id,messages,tokens_used,step,tools_executed)
            if crash_after_tool_number is not None and new_tools==crash_after_tool_number: raise SimulatedCrash('simulated crash after checkpoint')
    return {'status':'FALLBACK_MAX_ITERATIONS','final_answer':'FALLBACK_MAX_ITERATIONS: hand off to a human; step limit reached.','steps':steps,'llm_calls':llm_calls,'tokens_used':tokens_used,'tools_executed':tools_executed}
def run_agent(user_message,scenario_id,max_iterations=6,token_limit=6000,time_limit_seconds=10,run_id=None,crash_after_tool_number=None):
    run_id=run_id or f'{scenario_id}_{int(time.time()*1000)}'; messages=[{'role':'system','content':'You are an AGENT. Use tools to look up orders, consult policy, calculate refunds, and issue or escalate only according to the refund rules.'},{'role':'user','content':user_message}]
    return _continue(run_id,scenario_id,messages,0,0,[],max_iterations,token_limit,time_limit_seconds,time.monotonic(),crash_after_tool_number)
def resume_agent(run_id,max_iterations=6,token_limit=6000,time_limit_seconds=10):
    payload=json.loads((STATE/f'{run_id}.json').read_text(encoding='utf-8'))
    return _continue(run_id,payload['scenario_id'],payload['messages'],payload['tokens_used'],payload['step_number'],payload['tools_executed'],max_iterations,token_limit,time_limit_seconds,time.monotonic())
def _run_policy_eval():
    ev=pd.read_csv(ROOT/'data/raw/policy_eval.csv',keep_default_na=False); retr=Retriever(10); zero=Retriever(0); rows=[]; zrows=[]
    for _,r in ev.iterrows():
        ans,sources=retr.answer(r.question); hits=retr.retrieve(r.question,3)
        top_sources=[x['source'] for x in hits]
        rows.append({'question':r.question,'expected_doc':r.expected_doc,'retrieved_expected_doc':r.expected_doc in top_sources if r.expected_doc!='NONE' else False,'expected_fact_found':r.expected_fact in ans if r.expected_fact else False,'fallback':ans.startswith('I could not find'),'answer':ans,'sources':sources})
        ans0,_=zero.answer(r.question); zhits=zero.retrieve(r.question,3); zrows.append({'question':r.question,'expected_doc':r.expected_doc,'retrieved_expected_doc':r.expected_doc in [x['source'] for x in zhits] if r.expected_doc!='NONE' else False,'expected_fact_found':r.expected_fact in ans0 if r.expected_fact else False,'fallback':ans0.startswith('I could not find'),'answer':ans0})
    pd.DataFrame(rows).to_csv(OUT/'policy_eval_overlap10.csv',index=False); pd.DataFrame(zrows).to_csv(OUT/'policy_eval_overlap0.csv',index=False)
    return {'overlap10':{'expected_doc_hits':int(sum(x['retrieved_expected_doc'] for x in rows[:12])),'expected_fact_hits':int(sum(x['expected_fact_found'] for x in rows[:12])),'unanswerable_fallbacks':int(sum(x['fallback'] for x in rows[12:]))},'overlap0':{'expected_doc_hits':int(sum(x['retrieved_expected_doc'] for x in zrows[:12])),'expected_fact_hits':int(sum(x['expected_fact_found'] for x in zrows[:12])),'unanswerable_fallbacks':int(sum(x['fallback'] for x in zrows[12:])),'failures':[x['question'] for x in zrows[:12] if not x['expected_fact_found']]}}
def run():
    if (OUT/'refund_ledger.jsonl').exists(): (OUT/'refund_ledger.jsonl').unlink()
    if (OUT/'agent_runs.jsonl').exists(): (OUT/'agent_runs.jsonl').unlink()
    retr=Retriever(10); mock_llm.reset(); policy_metrics=_run_policy_eval()
    scenarios=pd.read_csv(ROOT/'data/raw/agent_scenarios.csv',keep_default_na=False); records=[]
    for _,s in scenarios.iterrows():
        result=run_agent(s.message,s.scenario_id,run_id=f"{s.scenario_id}_main")
        order_match=re.search(r'ORD-\d{6}',s.message); amount=0.0
        if result['status'] in ('REFUND','ESCALATE'):
            m=re.search(r'INR ([0-9.]+)',result['final_answer']); amount=float(m.group(1)) if m else 0.0
            if result['status']=='ESCALATE':
                amount=next((float(st['observation']['refund_inr']) for st in result['steps'] if st['tool']=='calculate_refund'),amount)
        elif result['status']=='FALLBACK_MAX_ITERATIONS': amount=0.0
        rec={'scenario_id':s.scenario_id,'status':result['status'],'final_answer':result['final_answer'],'refund_amount':amount,'llm_calls':result['llm_calls'],'tokens_used':result['tokens_used'],'steps':result['steps'],'tools_executed':result['tools_executed']}
        records.append(rec)
        with (OUT/'agent_runs.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(rec,default=str)+'\n')
    # Safe-stop demonstrations.
    s1=scenarios.iloc[0]; token_demo=run_agent(s1.message,'S1_token_demo',token_limit=200,run_id='S1_token_demo')
    timeout_demo=run_agent(s1.message,'S1_timeout_demo',time_limit_seconds=0,run_id='S1_timeout_demo')
    # Crash after 2 tools; resume from saved state. Ledger entry from the first two tools does not issue a refund.
    try: run_agent(s1.message,'S1_crash_resume',run_id='S1_crash_resume',crash_after_tool_number=2)
    except SimulatedCrash: pass
    resumed=resume_agent('S1_crash_resume')
    # Persist run details for README generation.
    metrics={'policy_eval':policy_metrics,'scenarios':records,'token_demo_status':token_demo['status'],'timeout_demo_status':timeout_demo['status'],'crash_resume':{'status':resumed['status'],'final_answer':resumed['final_answer'],'tools_executed':resumed['tools_executed'],'llm_calls':resumed['llm_calls'],'tokens_used':resumed['tokens_used']}}
    (OUT/'part_d_metrics.json').write_text(json.dumps(metrics,indent=2,default=str),encoding='utf-8')
    print('Part D complete:',json.dumps({'policy_eval':policy_metrics,'scenario_statuses':{r['scenario_id']:r['status'] for r in records},'token_demo':token_demo['status'],'timeout_demo':timeout_demo['status'],'resume':metrics['crash_resume']['status']}))
if __name__=='__main__': run()
