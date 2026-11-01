import {estimateFromIBMData,validateProfile,validateWorkplaceProfile} from './data';
const base=import.meta.env.VITE_API_BASE_URL?.replace(/\/$/,'');
export const isDemo=false;
export async function predict(profile,signal,rows) {
 const errors=validateProfile(profile);if(Object.keys(errors).length)throw new Error('Please check the highlighted profile fields.');
 if(!base)return predictWithCard('/data/ibm-attrition.json',profile,signal);
 const response=await fetch(`${base}/predict`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(profile),signal});
 if(!response.ok)throw new Error('The prediction service rejected this profile or is unavailable. Check the fields and confirm that the service uses the current 20-feature schema. Your profile is preserved.');
 const result=await response.json();if(typeof result.probability!=='number'||result.probability<0||result.probability>1)throw new Error('The service returned an invalid probability. Please try again.');
 return {...result,mode:'model',model_type:result.model_type||null};
}

export async function predictWorkplace(profile,signal){
 const errors=validateWorkplaceProfile(profile);
 if(Object.keys(errors).length)throw new Error('Please check the highlighted workplace profile fields.');
 if(!base)return predictWithCard('/data/workplace-turnover.json',profile,signal);
 const response=await fetch(`${base}/workplace/predict`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(profile),signal});
 if(!response.ok){let detail='The workplace prediction service is unavailable.';try{const body=await response.json();detail=body.detail||detail;}catch{}throw new Error(detail);}
 const result=await response.json();
 if(typeof result.probability!=='number'||result.probability<0||result.probability>1)throw new Error('The service returned an invalid probability.');
 return {...result,mode:'model'};
}

async function readCard(path,signal){const response=await fetch(path,{signal});if(!response.ok)throw new Error(`Could not load trained model card (${response.status}).`);return response.json();}
async function predictWithCard(path,profile,signal){
 const card=await readCard(path,signal);let logit=card.baseline_log_odds;const contributions=[];
 for(const [key,entry] of Object.entries(card.numeric)){const value=Number(profile[key]);const contribution=((value-entry.mean)/entry.scale)*entry.coefficient;logit+=contribution;contributions.push({feature:key,value,contribution});}
 for(const [key,entry] of Object.entries(card.categorical)){const value=String(profile[key]??entry.impute);const contribution=(entry.weights[value]??0)-entry.expected_weight;logit+=contribution;contributions.push({feature:key,value,contribution});}
 const probability=1/(1+Math.exp(-Math.max(-35,Math.min(35,logit))));
 contributions.sort((a,b)=>Math.abs(b.contribution)-Math.abs(a.contribution));
 return {probability,mode:'model-client',model_type:card.model_type,model_version:card.model_version,target:card.target,training_entries:card.training_entries,dataset_entries:card.dataset_entries,explanation:{method:'logistic-regression contributions',output:'log-odds',base_value:card.baseline_log_odds,contributions:contributions.slice(0,8)}};
}

async function get(path,signal){
 if(!base){const card=await readCard(path.includes('workplace')?'/data/workplace-turnover.json':'/data/ibm-attrition.json',signal);if(path.endsWith('/model'))return {model_id:card.model_id,model_version:card.model_version,model_type:card.model_type,target:card.target,features:card.features,training_data:{n:card.dataset_entries,n_train:card.training_entries,n_test:card.test_entries},warning:card.warning};return {model_id:card.model_id,model_version:card.model_version,model_type:card.model_type,seed:card.seed,n_total:card.dataset_entries,n_train:card.training_entries,n_test:card.test_entries,threshold:card.threshold,held_out_test:card.metrics};}
 const response=await fetch(`${base}${path}`,{signal});
 if(!response.ok)throw new Error(`Could not load model details (${response.status}).`);
 return response.json();
}
export const getModelInfo=signal=>get('/model',signal);
export const getEvaluation=signal=>get('/evaluation',signal);
export const getWorkplaceModelInfo=signal=>get('/workplace/model',signal);
export const getWorkplaceEvaluation=signal=>get('/workplace/evaluation',signal);
