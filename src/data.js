export const roles = ['Sales Executive','Research Scientist','Laboratory Technician','Manufacturing Director','Healthcare Representative','Manager','Sales Representative','Research Director','Human Resources'];
export const departments = ['Human Resources','Research & Development','Sales'];
// Valid combinations observed in the IBM sample schema. Manager occurs in
// both Sales and R&D; the other roles belong to one department.
export const rolesByDepartment = {
  'Human Resources':['Human Resources','Manager'],
  'Research & Development':['Healthcare Representative','Laboratory Technician','Manager','Manufacturing Director','Research Director','Research Scientist'],
  'Sales':['Manager','Sales Executive','Sales Representative'],
};
export const travelOptions = ['Non-Travel','Travel_Rarely','Travel_Frequently'];
export const maritalStatuses = ['Divorced','Married','Single'];
export const initialProfile = {
  JobRole:'Sales Executive', Department:'Sales', BusinessTravel:'Travel_Rarely', MaritalStatus:'Single',
  OverTime:'Yes', Age:32, JobLevel:1, MonthlyIncome:4000, JobSatisfaction:2, WorkLifeBalance:2,
  EnvironmentSatisfaction:3, JobInvolvement:3, YearsAtCompany:3, YearsSinceLastPromotion:1,
  YearsWithCurrManager:2, YearsInCurrentRole:2, TotalWorkingYears:8, StockOptionLevel:0,
  NumCompaniesWorked:2, DistanceFromHome:24,
};
export const workplaceDepartments = ['IT','RandD','accounting','hr','management','marketing','product_mng','sales','support','technical'];
export const workplaceSalaries = ['low','medium','high'];
export const initialWorkplaceProfile = {
  satisfaction_level:0.62, last_evaluation:0.72, number_project:4,
  average_montly_hours:180, time_spend_company:3, Work_accident:0,
  promotion_last_5years:0, Department:'sales', salary:'medium',
};
export const WORKPLACE_FEATURES = Object.keys(initialWorkplaceProfile);
export const PROFILE_FEATURES = Object.keys(initialProfile);
export const MILES_PER_KM = 1 / 1.609344;
export function parseCSV(csv) {
  const records=[];let row=[],field='',quoted=false;
  const input=csv.replace(/^\uFEFF/,'');
  for(let i=0;i<input.length;i++){
    const ch=input[i];
    if(ch==='"'){
      if(quoted&&input[i+1]==='"'){field+='"';i++;}
      else quoted=!quoted;
    } else if(ch===','&&!quoted){row.push(field);field='';}
    else if((ch==='\n'||ch==='\r')&&!quoted){
      if(ch==='\r'&&input[i+1]==='\n')i++;
      row.push(field);field='';if(row.some(value=>value!==''))records.push(row);row=[];
    } else field+=ch;
  }
  if(field!==''||row.length){row.push(field);records.push(row);}
  const headers=records.shift()||[];
  return records.map(values=>Object.fromEntries(headers.map((header,index)=>{const value=values[index]??'';return [header,value!==''&&!Number.isNaN(Number(value))?Number(value):value];})));
}
export function normalizeWorkplaceRows(rows){return rows.map(row=>({...row,Attrition:Number(row.left)===1?'Yes':'No'}));}
export function summary(rows) {const left=rows.filter(r=>r.Attrition==='Yes').length; return {total:rows.length,left,stayed:rows.length-left,rate:rows.length?left/rows.length*100:0};}
export function groups(rows,key,values) {return values.map(label=>({label,...summary(rows.filter(r=>String(r[key])===String(label)))}));}
export function profileFromRow(row) { return Object.fromEntries(PROFILE_FEATURES.map(key=>[key,row[key]])); }
export function validateProfile(p) {
  const errors={};
  const ranges={Age:[18,60],JobLevel:[1,5],MonthlyIncome:[100,100000],JobSatisfaction:[1,4],WorkLifeBalance:[1,4],EnvironmentSatisfaction:[1,4],JobInvolvement:[1,4],YearsAtCompany:[0,40],YearsSinceLastPromotion:[0,40],YearsWithCurrManager:[0,40],YearsInCurrentRole:[0,40],TotalWorkingYears:[0,40],StockOptionLevel:[0,3],NumCompaniesWorked:[0,9],DistanceFromHome:[1,100]};
  for(const [key,[min,max]] of Object.entries(ranges)) if(p[key]===''||!Number.isFinite(Number(p[key]))||Number(p[key])<min||Number(p[key])>max||!Number.isInteger(Number(p[key]))) errors[key]=`Enter a whole number from ${min} to ${max}.`;
  if(Number(p.YearsSinceLastPromotion)>Number(p.YearsAtCompany)) errors.YearsSinceLastPromotion='Cannot exceed years at the company.';
  if(Number(p.YearsWithCurrManager)>Number(p.YearsAtCompany)) errors.YearsWithCurrManager='Cannot exceed years at the company.';
  if(Number(p.YearsInCurrentRole)>Number(p.YearsAtCompany)) errors.YearsInCurrentRole='Cannot exceed years at the company.';
  if(Number(p.TotalWorkingYears)<Number(p.YearsAtCompany)) errors.TotalWorkingYears='Cannot be less than years at the company.';
  if(!roles.includes(p.JobRole)) errors.JobRole='Choose a valid job role.';
  if(!departments.includes(p.Department)) errors.Department='Choose a valid department.';
  else if(!rolesByDepartment[p.Department]?.includes(p.JobRole)) errors.JobRole='Choose a role available in this department.';
  if(!travelOptions.includes(p.BusinessTravel)) errors.BusinessTravel='Choose a valid travel frequency.';
  if(!maritalStatuses.includes(p.MaritalStatus)) errors.MaritalStatus='Choose a valid marital status.';
  if(!['Yes','No'].includes(p.OverTime)) errors.OverTime='Choose Yes or No.';
  return errors;
}
export function validateWorkplaceProfile(p){
  const errors={};
  const ranges={satisfaction_level:[0.09,1],last_evaluation:[0.36,1],number_project:[2,7],average_montly_hours:[96,310],time_spend_company:[2,10],Work_accident:[0,1],promotion_last_5years:[0,1]};
  for(const [key,[min,max]] of Object.entries(ranges))if(p[key]===''||!Number.isFinite(Number(p[key]))||Number(p[key])<min||Number(p[key])>max||(['number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years'].includes(key)&&!Number.isInteger(Number(p[key]))))errors[key]=`Enter a valid value from ${min} to ${max}.`;
  if(!workplaceDepartments.includes(p.Department))errors.Department='Choose a department from the HRIS dataset.';
  if(!workplaceSalaries.includes(p.salary))errors.salary='Choose a salary band from the HRIS dataset.';
  return errors;
}
// Transparent, local fallback over the bundled fictional IBM sample. This is an
// exploratory similarity estimate, not a validated or calibrated employment model.
export function estimateFromIBMData(profile, rows, neighbors=50) {
  if(!rows?.length) throw new Error('The IBM sample dataset is not loaded.');
  const numeric=['Age','JobLevel','MonthlyIncome','JobSatisfaction','WorkLifeBalance','EnvironmentSatisfaction','JobInvolvement','YearsAtCompany','YearsSinceLastPromotion','YearsWithCurrManager','YearsInCurrentRole','TotalWorkingYears','StockOptionLevel','NumCompaniesWorked','DistanceFromHome'];
  const categorical=['JobRole','Department','BusinessTravel','MaritalStatus','OverTime'];
  const ranges={Age:42,JobLevel:4,MonthlyIncome:99900,JobSatisfaction:3,WorkLifeBalance:3,EnvironmentSatisfaction:3,JobInvolvement:3,YearsAtCompany:40,YearsSinceLastPromotion:40,YearsWithCurrManager:17,YearsInCurrentRole:18,TotalWorkingYears:40,StockOptionLevel:3,NumCompaniesWorked:9,DistanceFromHome:28};
  const weightedDistance=row=>{let sum=0,weight=0;for(const key of numeric){const w=key==='MonthlyIncome'?0.7:1;sum+=w*Math.min(1,Math.abs(Number(profile[key])-Number(row[key]))/ranges[key]);weight+=w;}for(const key of categorical){const w=key==='JobRole'||key==='OverTime'?1.2:0.8;sum+=w*(profile[key]===row[key]?0:1);weight+=w;}return sum/weight;};
  const nearest=rows.map(row=>({row,distance:weightedDistance(row)})).sort((a,b)=>a.distance-b.distance).slice(0,Math.min(neighbors,rows.length));
  const base=summary(rows).rate/100,kernel=nearest.map(({distance})=>1/Math.pow(0.04+distance,2)),effective=kernel.reduce((a,b)=>a+b,0);
  const neighborRate=nearest.reduce((sum,item,i)=>sum+kernel[i]*(item.row.Attrition==='Yes'?1:0),0)/effective;
  const probability=Math.max(0,Math.min(1,(neighborRate*nearest.length+base*12)/(nearest.length+12)));
  const avgDistance=nearest.reduce((sum,item,i)=>sum+kernel[i]*item.distance,0)/effective;
  return {probability:Number(probability.toFixed(4)),mode:'sample-estimate',model_version:'ibm-knn-1',explanation:null,similarity:{matched:nearest.length,meanDistance:Number(avgDistance.toFixed(3)),datasetSize:rows.length,baseRate:Number(base.toFixed(4))}};
}
// Used only by the separate what-if illustration, never by the prediction form.
export function demoProbability(p) { const score=15+(p.OverTime==='Yes'?23:0)+(4-Number(p.JobSatisfaction))*8+(4-Number(p.WorkLifeBalance))*5+(Number(p.MonthlyIncome)<4500?7:0)+(Number(p.YearsAtCompany)<4?5:0)+Number(p.DistanceFromHome)*.15+Math.min(8,Number(p.YearsSinceLastPromotion)); return Math.round(Math.max(5,Math.min(92,score))); }
export function riskBand(n) {return n>=60?'Elevated':n>=30?'Moderate':'Lower';}
