import {estimateFromIBMData,validateProfile} from './data';
const base=import.meta.env.VITE_API_BASE_URL?.replace(/\/$/,'');
export const isDemo=!base;
export async function predict(profile,signal,rows) {
 const errors=validateProfile(profile);if(Object.keys(errors).length)throw new Error('Please check the highlighted profile fields.');
 if(!base){await new Promise((resolve,reject)=>{const timer=setTimeout(resolve,250);signal?.addEventListener('abort',()=>{clearTimeout(timer);reject(new DOMException('Aborted','AbortError'));},{once:true});});return estimateFromIBMData(profile,rows);}
 const response=await fetch(`${base}/predict`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(profile),signal});
 if(!response.ok)throw new Error('The prediction service is unavailable or needs the updated 15-feature model. Your profile is preserved. Please try again.');
 const result=await response.json();if(typeof result.probability!=='number'||result.probability<0||result.probability>1)throw new Error('The service returned an invalid probability. Please try again.');
 return {...result,mode:'model'};
}
