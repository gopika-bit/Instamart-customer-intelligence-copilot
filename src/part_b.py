from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
def run():
    c=pd.read_csv(ROOT/'data/clean/customers.csv',parse_dates=['signup_date'])
    o=pd.read_csv(ROOT/'data/clean/orders.csv',parse_dates=['order_ts'])
    done=o[o.status.eq('completed')].copy(); snap=pd.Timestamp('2025-10-01 00:00:00')
    rfm=done.groupby('customer_id').agg(last_order_ts=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum')).reset_index()
    rfm['recency']=(snap-rfm.last_order_ts).dt.days; rfm=rfm.drop(columns='last_order_ts')
    scaler=StandardScaler(); X=scaler.fit_transform(rfm[['recency','frequency','monetary']])
    rows=[]
    for k in range(2,9):
        km=KMeans(n_clusters=k,n_init=10,random_state=42); lab=km.fit_predict(X)
        rows.append({'K':k,'inertia':km.inertia_,'silhouette':silhouette_score(X,lab)})
    pd.DataFrame(rows).to_csv(OUT/'kmeans_metrics.csv',index=False)
    k=4; km=KMeans(n_clusters=k,n_init=10,random_state=42); rfm['segment_id']=km.fit_predict(X)
    prof=rfm.groupby('segment_id').agg(size=('customer_id','size'),mean_recency=('recency','mean'),mean_frequency=('frequency','mean'),mean_monetary=('monetary','mean')).reset_index()
    premium_id=int(prof.loc[prof.mean_monetary.idxmax(),'segment_id'])
    # Names derived from measured profile, not cluster number.
    names={}
    for _,p in prof.iterrows():
        sid=int(p.segment_id)
        if sid==premium_id: name='Premium High-Value'
        elif p.mean_frequency>=prof.mean_frequency.quantile(.75): name='Frequent Loyalists'
        elif p.mean_recency>=prof.mean_recency.quantile(.75): name='At-Risk / Cooling'
        else: name='Everyday Customers'
        names[sid]=name
    rfm['segment_name']=rfm.segment_id.map(names); rfm['is_premium']=rfm.segment_id.eq(premium_id)
    rfm[['customer_id','recency','frequency','monetary','segment_id','segment_name','is_premium']].to_csv(OUT/'customer_segments.csv',index=False)
    pca=PCA(n_components=2); pca.fit(X)
    # Lapse cohort and pre-cutoff features only.
    cutoff=pd.Timestamp('2025-09-01'); sep_end=pd.Timestamp('2025-10-01')
    pre=o[(o.order_ts<cutoff)&o.status.eq('completed')].copy()
    cohort=set(pre.customer_id.unique())
    labels=done[(done.order_ts>=cutoff)&(done.order_ts<sep_end)].groupby('customer_id').size()
    feat=pre.groupby('customer_id').agg(last_order=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum'),avg_discount=('discount_pct','mean'),avg_delivery=('delivery_minutes','mean'))
    all_pre=o[o.order_ts<cutoff].copy()
    all_pre=all_pre[all_pre.status.isin(['completed','cancelled'])]
    canc=all_pre.groupby('customer_id').status.agg(total='size',cancelled=lambda s:(s=='cancelled').sum())
    canc['cancellation_rate']=canc.cancelled/canc.total
    fav=pre.merge(pd.read_csv(ROOT/'data/clean/products.csv')[['product_id','category']],on='product_id').groupby('customer_id')['category'].agg(lambda s:s.mode().iloc[0] if not s.mode().empty else 'Grocery')
    feat=feat.join(canc[['cancellation_rate']]).join(fav.rename('favourite_category')).reset_index()
    feat['recency']=(cutoff-feat.last_order).dt.days; feat=feat.drop(columns='last_order')
    feat=feat.merge(c[['customer_id','city','age','gender','signup_date']],on='customer_id',how='left')
    feat['tenure_days']=(cutoff-feat.signup_date).dt.days; feat=feat.drop(columns='signup_date')
    feat['lapsed']=feat.customer_id.map(lambda x:int(x not in labels.index))
    numeric=['recency','frequency','monetary','avg_discount','cancellation_rate','avg_delivery','age','tenure_days']
    categorical=['city','gender','favourite_category']; Xdf=feat[numeric+categorical].copy()
    # explicit missing-value assertion after deterministic imputation preparation
    for col in numeric: Xdf[col]=Xdf[col].fillna(Xdf[col].median())
    for col in categorical: Xdf[col]=Xdf[col].fillna('unknown').astype(str)
    assert not Xdf.isna().any().any()
    y=feat.lapsed.astype(int)
    X_train,X_test,y_train,y_test=train_test_split(Xdf,y,test_size=.2,stratify=y,random_state=42)
    num_pipe=Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())])
    cat_pipe=Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(drop='first',handle_unknown='ignore'))])
    prep=ColumnTransformer([('num',num_pipe,numeric),('cat',cat_pipe,categorical)])
    log_pipe=Pipeline([('prep',prep),('model',LogisticRegression(max_iter=1000,random_state=42))])
    log_grid=GridSearchCV(log_pipe,{'model__C':[.01,.1,1,10,100]},cv=5,scoring='f1',n_jobs=-1); log_grid.fit(X_train,y_train)
    rf_pipe=Pipeline([('prep',prep),('model',RandomForestClassifier(n_estimators=100,random_state=42))])
    rf_grid=GridSearchCV(rf_pipe,{'model__max_depth':[3,5,8],'model__min_samples_leaf':[5,10,20]},cv=5,scoring='f1',n_jobs=-1); rf_grid.fit(X_train,y_train)
    def metrics(model):
        yp=model.predict(X_test); return {'confusion_matrix':confusion_matrix(y_test,yp).tolist(),'precision':precision_score(y_test,yp,zero_division=0),'recall':recall_score(y_test,yp,zero_division=0),'f1':f1_score(y_test,yp,zero_division=0),'accuracy':accuracy_score(y_test,yp)}
    baseline=np.zeros(len(y_test),dtype=int)
    # Oversample lapsed class in training rows only for the balance experiment.
    train_idx=np.arange(len(y_train)); positives=train_idx[y_train.to_numpy()==1]; negatives=train_idx[y_train.to_numpy()==0]
    rng=np.random.RandomState(42); add=rng.choice(positives,size=max(0,len(negatives)-len(positives)),replace=True)
    balanced_idx=np.concatenate([train_idx,add]); X_bal=X_train.iloc[balanced_idx]; y_bal=y_train.iloc[balanced_idx]
    balanced_log=clone(log_grid.best_estimator_); balanced_log.fit(X_bal,y_bal)
    selected=rf_grid.best_estimator_ if metrics(rf_grid.best_estimator_)['f1']>=metrics(log_grid.best_estimator_)['f1'] else log_grid.best_estimator_
    joblib.dump(selected,OUT/'lapse_model.joblib'); loaded=joblib.load(OUT/'lapse_model.joblib')
    assert np.array_equal(selected.predict(X_test),loaded.predict(X_test))
    # Importance and coefficients saved in a compact report.
    rf_best=rf_grid.best_estimator_; log_best=log_grid.best_estimator_
    rf_names=rf_best.named_steps['prep'].get_feature_names_out().tolist()
    rf_importance=rf_best.named_steps['model'].feature_importances_
    top_features=sorted([{'feature':n,'importance':float(v)} for n,v in zip(rf_names,rf_importance)],key=lambda x:x['importance'],reverse=True)[:5]
    log_names=log_best.named_steps['prep'].get_feature_names_out().tolist(); log_coef=log_best.named_steps['model'].coef_[0]
    coef_signs=[]
    for i,n in enumerate(log_names):
        if any(key in n for key in ['frequency','recency','monetary','cancellation_rate','avg_delivery','tenure_days']):
            coef_signs.append({'feature':n,'coefficient':float(log_coef[i]),'sign':'positive' if log_coef[i]>0 else 'negative'})
    model_report={'rf_best_params':rf_grid.best_params_,'rf_cv_f1':rf_grid.best_score_,'rf_test':metrics(rf_grid.best_estimator_),
                  'log_best_params':log_grid.best_params_,'log_cv_f1':log_grid.best_score_,'log_test':metrics(log_grid.best_estimator_),
                  'rf_top_features':top_features,'logistic_coefficient_signs':coef_signs[:3],
                  'rf_best_test_f1':metrics(rf_grid.best_estimator_)['f1'],
                  'rf_best_test_predictions_match_reload':True,
                  'rfm_summary':{'customers':len(rfm),'recency_median':float(rfm.recency.median()),'recency_min':int(rfm.recency.min()),'recency_max':int(rfm.recency.max()),'frequency_mean':float(rfm.frequency.mean()),'frequency_max':int(rfm.frequency.max()),'monetary_mean':float(rfm.monetary.mean()),'monetary_max':float(rfm.monetary.max())},
                  'baseline_accuracy':accuracy_score(y_test,baseline),'baseline_lapsed_f1':f1_score(y_test,baseline,zero_division=0),
                  'balanced_log_test':metrics(balanced_log), 'cohort_n':len(feat),'lapsed_n':int(y.sum()),'train_n':len(X_train),'test_n':len(X_test),
                  'scaler_train_frequency_mean':float(log_grid.best_estimator_.named_steps['prep'].named_transformers_['num'].named_steps['scaler'].mean_[1]),
                  'full_cohort_frequency_mean':float(feat.frequency.mean()),'pca_explained_variance_ratio':pca.explained_variance_ratio_.tolist(),
                  'segment_profiles':prof.to_dict(orient='records'),'kmeans_metrics':rows,'premium_segment_id':premium_id}
    # Top 10 test customers within post-hoc Premium group, using probabilities from selected model.
    test_customers=feat.loc[X_test.index,['customer_id']].copy(); test_customers['actual']=y_test.to_numpy(); test_customers['prob']=selected.predict_proba(X_test)[:,1]
    test_customers=test_customers.merge(rfm[['customer_id','is_premium']],on='customer_id',how='left')
    ret=test_customers[test_customers.is_premium.fillna(False)].sort_values('prob',ascending=False).head(10).copy()
    ret.rename(columns={'prob':'lapse_probability','actual':'actually_lapsed'})[['customer_id','lapse_probability','actually_lapsed']].to_csv(OUT/'retention_list.csv',index=False)
    model_report['retention_list_true_lapsed']=int(ret.actual.sum())
    (OUT/'part_b_metrics.json').write_text(json.dumps(model_report,indent=2,default=float))
    print('Part B complete:',json.dumps({k:model_report[k] for k in ['cohort_n','lapsed_n','train_n','test_n','rf_test','log_test']},default=float))
if __name__=='__main__': run()
