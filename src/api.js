import {demoProbability,validateProfile} from './data';
const base=import.meta.env.VITE_API_BASE_URL?.replace(/\/$/,'');
export const isDemo=!base;
export async function predict(profile,signal) {
 const errors=validateProfile(profile);if(Object.keys(errors).length)throw new Error('Please check the highlighted profile fields.');
 if(!base){await new Promise(resolve=>setTimeout(resolve,650));return {probability:demoProbability(profile)/100,mode:'illustrative',model_version:null,explanation:null};}
 const response=await fetch(`${base}/predict`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(profile),signal});
 if(!response.ok)throw new Error('The prediction service is unavailable. Your profile is preserved. Please try again.');
 const result=await response.json();if(typeof result.probability!=='number'||result.probability<0||result.probability>1)throw new Error('The service returned an invalid probability. Please try again.');
 return {...result,mode:'model'};
}
