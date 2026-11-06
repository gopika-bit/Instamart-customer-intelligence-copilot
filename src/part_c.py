from pathlib import Path
import re, json, random, time
import pandas as pd
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import mock_llm
OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
EMAIL=re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+'); PHONE=re.compile(r'\b[6-9]\d{9}\b'); ORDER=re.compile(r'ORD-\d{6}')
ALLOWED={'billing','delivery','technical','fraud'}; MAX_INPUT_TOKENS=400; BASE_DELAY=.05
ZERO='You are a CLASSIFY support-ticket classifier. Allowed categories: billing, delivery, technical, fraud. Reply with JSON only in exactly this form: {"category": "...", "reason": "..."}. The category must be one of the four allowed categories.'
FEW='You are a CLASSIFY support-ticket classifier. Allowed categories: billing, delivery, technical, fraud. Reply with JSON only in exactly this form: {"category": "...", "reason": "..."}. The category must be one of the four allowed categories.\nEXAMPLES:\nTicket: Unauthorized transaction on my card.\n{"category":"fraud","reason":"unauthorized transaction"}\nTicket: The app crashes and login fails.\n{"category":"technical","reason":"app crash and login"}'
def redact(text):
    counts={}
    text,n=EMAIL.subn('[EMAIL]',text); counts['emails']=n
    text,n=PHONE.subn('[PHONE]',text); counts['phones']=n
    text,n=ORDER.subn('[ORDER_ID]',text); counts['order_ids']=n
    return text,counts
def classify_run(prompt, budget=None, segments=None, output_path=None):
    tickets=pd.read_csv(ROOT/'data/raw/tickets.csv',keep_default_na=False)
    segments=segments if segments is not None else pd.read_csv(OUT/'customer_segments.csv')
    seg=segments.set_index('customer_id').to_dict('index')
    mock_llm.reset(); random.seed(42); total_tokens=0; results=[]; category_counts={}; redact_counts={'emails':0,'phones':0,'order_ids':0}; retries=[]
    for _,t in tickets.iterrows():
        text=str(t['text']); tid=t['ticket_id']; status=''; category=''; used=0; redacted_text=''
        if not text.strip(): status='skipped_empty'
        elif mock_llm.est_tokens(text)>MAX_INPUT_TOKENS: status='flagged_long'
        else:
            red,counts=redact(text); redacted_text=red
            for k,v in counts.items(): redact_counts[k]+=v
            assert not EMAIL.search(red) and not PHONE.search(red) and not ORDER.search(red), f'PII remained for {tid}'
            if budget is not None and total_tokens>=budget: status='budget_exhausted'
            else:
                messages=[{'role':'system','content':prompt},{'role':'user','content':red}]
                attempt=0
                while True:
                    if budget is not None and total_tokens>=budget:
                        status='budget_exhausted'; break
                    try:
                        response=mock_llm.chat(messages)
                        used=response.usage.total_tokens; total_tokens+=used
                        raw=response.choices[0].message.content or ''
                        try:
                            obj=json.loads(raw)
                            if isinstance(obj,dict) and obj.get('category') in ALLOWED:
                                category=obj['category']; status='classified_json'
                            else: status='flagged_invalid_output'
                        except (json.JSONDecodeError,TypeError):
                            words=re.findall(r'[a-z]+',raw.lower()); matches=[w for w in words if w in ALLOWED]
                            if len(set(matches))==1:
                                category=matches[0]; status='classified_keyword'
                            else: status='flagged_invalid_output'
                        break
                    except mock_llm.TransientLLMError:
                        if attempt>=3: status='flagged_invalid_output'; break
                        wait=BASE_DELAY*(2**attempt)*random.uniform(.9,1.15)
                        print(f'RETRY ticket={tid} retry={attempt} wait={wait:.4f}s')
                        retries.append({'ticket_id':tid,'retry':attempt,'wait':wait}); time.sleep(wait); attempt+=1
        s=seg.get(t['customer_id'],{}); segment_name=s.get('segment_name','')
        premium=bool(s.get('is_premium',False))
        if status in {'skipped_empty','flagged_long','flagged_invalid_output','budget_exhausted'}: route='human_review'; priority='NORMAL'
        elif category=='fraud': route='security_team'; priority='HIGH'
        elif premium: route='premium_desk'; priority='HIGH'
        else: route='support_queue'; priority='NORMAL'
        results.append({'ticket_id':tid,'customer_id':t['customer_id'],'status':status,'category':category,'segment_name':segment_name,'priority':priority,'route':route,'tokens_used':used,'true_category':t['true_category'],'redacted_text':red})
    df=pd.DataFrame(results)
    if output_path: df[['ticket_id','customer_id','status','category','segment_name','priority','route','tokens_used']].to_csv(output_path,index=False)
    classified=df[df.status.isin(['classified_json','classified_keyword'])]
    accuracy=float((classified.category==classified.true_category).mean()) if len(classified) else 0
    summary={'tickets_total':len(df),'classified_json':int((df.status=='classified_json').sum()),'classified_keyword':int((df.status=='classified_keyword').sum()),'flagged_invalid_output':int((df.status=='flagged_invalid_output').sum()),'skipped_empty':int((df.status=='skipped_empty').sum()),'flagged_long':int((df.status=='flagged_long').sum()),'budget_exhausted':int((df.status=='budget_exhausted').sum()),'classified':len(classified),'accuracy':accuracy,'tokens_total':total_tokens,'tokens_per_classified':total_tokens/len(classified) if len(classified) else 0,'redactions':redact_counts,'redactions_total':sum(redact_counts.values()),'retries':len(retries),'confusion':pd.crosstab(classified.true_category,classified.category).to_dict()}
    return df,summary

def run():
    seg=pd.read_csv(OUT/'customer_segments.csv')
    zero,zsum=classify_run(ZERO,None,seg)
    few,fsum=classify_run(FEW,None,seg)
    # Token-budget demonstration; use a documented budget that reliably exhausts.
    _,bsum=classify_run(ZERO,5000,seg,OUT/'triage_budget_demo.csv')
    final,finalsum=classify_run(FEW,100000,seg,OUT/'triage_results.csv')
    (OUT/'part_c_metrics.json').write_text(json.dumps({'zero_shot':zsum,'few_shot':fsum,'budget_demo':bsum,'final_few_shot':finalsum},indent=2,default=str))
    print('Part C metrics:',json.dumps({'zero_shot':zsum,'few_shot':fsum,'budget_demo':bsum},default=str))
if __name__=='__main__': run()
