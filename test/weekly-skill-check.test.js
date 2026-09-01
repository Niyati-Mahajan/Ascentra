const assert=require('assert');
const fs=require('fs');
const path=require('path');
const Core=require('../career-core');

const fullstack=Core.roleById('fullstack');
const baseProfile={
  student:{
    target:'fullstack',
    department:'CSE-AI & ML',
    cgpa:9.2,
    backlogs:0,
    skills:{JavaScript:28,React:42,'Node.js':62,'Express.js':69,SQL:61,'REST APIs':69},
    projects:[],
    internships:0,
    certifications:[],
    roadmap:{}
  },
  resume:{parsed:{text:'Projects JavaScript SQL Git GitHub',skills:['JavaScript','SQL','Git'],projects:[],sections:{projects:true}}},
  learning:{quizHistory:[],lastQuiz:null}
};

function withQuiz(record){
  const profile=structuredClone(baseProfile);
  profile.learning={quizHistory:[record],lastQuiz:record};
  return profile;
}

const appSource=fs.readFileSync(path.join(__dirname,'..','app.js'),'utf8');
const activeSubmit=appSource.match(/document\.addEventListener\('submit',e=>\{if\(e\.target\.id!=='weekly-skill-form'\)[\s\S]*?render\(\)\},true\);/)?.[0]||'';
assert(activeSubmit.includes('perSkillScores=Object.fromEntries'),'new Weekly Check attempts must store per-skill scores');
assert(activeSubmit.includes('questions=qs.map'),'new Weekly Check attempts must store question provenance');
assert(activeSubmit.includes('correct:ok'),'new question provenance must include per-skill correctness');
assert(activeSubmit.includes('compatibilityBefore=roleScore(r)'),'Weekly Check must capture canonical compatibility before submit');
assert(activeSubmit.includes('compatibilityAfter=Core.roleScore(r,afterCtx)'),'Weekly Check must calculate canonical compatibility after submit');
assert(!activeSubmit.includes('quizAdjustment='),'active Weekly Check submit path must not write quizAdjustment');
assert(appSource.includes("'ASCENTRA / '+viewTitle()"),'top page label should use the friendly Weekly Skill Check title');
assert(appSource.includes("view==='quiz'?'Weekly skill check.'"),'Weekly Check should keep a single main heading');
assert(appSource.includes("hasEvidenceChanges?'Evidence strengthened':'Assessment evidence'"),'legacy attempts should label shared assessment evidence without fabricating evidence changes');
assert(appSource.includes('Detailed per-skill results were not stored for this earlier check.'),'legacy attempts must not invent per-skill correctness');

const legacy=withQuiz({score:67,correct:2,total:3,skills:['REST APIs','React','DSA']});
assert.equal(Core.getSkillEvidence('React',fullstack,legacy).assessmentEvidence.score,55,'legacy shared scores must still be capped at sharedAssessment');
assert.equal(Core.getSkillEvidence('DSA',fullstack,legacy).evidenceScore,55,'legacy records must preserve current shared-score fallback');

const jsReactCorrectNodeWrong=withQuiz({
  score:67,
  correct:2,
  total:3,
  skills:['JavaScript','React','Node.js'],
  perSkillScores:{JavaScript:100,React:100,'Node.js':0},
  questions:[
    {id:'javascript-promise',skill:'JavaScript',correct:true},
    {id:'react-state',skill:'React',correct:true},
    {id:'node-js-usage',skill:'Node.js',correct:false}
  ]
});
assert.equal(Core.getSkillEvidence('JavaScript',fullstack,jsReactCorrectNodeWrong).assessmentEvidence.score,55,'correct per-skill answer should create bounded assessment evidence');
assert.equal(Core.getSkillEvidence('Node.js',fullstack,jsReactCorrectNodeWrong).assessmentEvidence.score,0,'wrong per-skill answer must not receive shared quiz evidence');
assert(Core.getSkillEvidence('JavaScript',fullstack,jsReactCorrectNodeWrong).evidenceScore>Core.getSkillEvidence('JavaScript',fullstack,baseProfile).evidenceScore,'correct JavaScript answer should strengthen JavaScript evidence');
assert.equal(Core.getSkillEvidence('Node.js',fullstack,jsReactCorrectNodeWrong).evidenceScore,Core.getSkillEvidence('Node.js',fullstack,baseProfile).evidenceScore,'wrong Node.js answer should not strengthen Node.js evidence');

const nodeCorrectOnly=withQuiz({
  score:33,
  correct:1,
  total:3,
  skills:['JavaScript','React','Node.js'],
  perSkillScores:{JavaScript:0,React:0,'Node.js':100},
  questions:[
    {id:'javascript-promise',skill:'JavaScript',correct:false},
    {id:'react-state',skill:'React',correct:false},
    {id:'node-js-usage',skill:'Node.js',correct:true}
  ]
});
assert.equal(Core.getSkillEvidence('JavaScript',fullstack,nodeCorrectOnly).assessmentEvidence.score,0,'wrong JavaScript answer should not create JavaScript assessment evidence');
assert.equal(Core.getSkillEvidence('React',fullstack,nodeCorrectOnly).assessmentEvidence.score,0,'wrong React answer should not create React assessment evidence');
assert.equal(Core.getSkillEvidence('Node.js',fullstack,nodeCorrectOnly).assessmentEvidence.score,55,'correct Node.js answer should create bounded Node.js assessment evidence');

const onlyDsaCorrect={
  student:{target:'fullstack',department:'CSE',cgpa:9.2,backlogs:0,skills:{},projects:[],internships:0,certifications:[],roadmap:{}},
  resume:{parsed:{text:'',skills:[],projects:[],sections:{}}},
  learning:{quizHistory:[],lastQuiz:{score:100,correct:1,total:1,skills:['DSA'],perSkillScores:{DSA:100},questions:[{id:'dsa-hash-map',skill:'DSA',correct:true}]}}
};
assert.equal(Core.getSkillEvidence('DSA',fullstack,onlyDsaCorrect).evidenceScore,55,'one per-skill quiz answer must remain capped as supporting evidence');
assert.equal(Core.getSkillEvidence('DSA',fullstack,onlyDsaCorrect).status,'Developing','one per-skill quiz answer should not automatically prove readiness for DSA');

const laterWrong=structuredClone(baseProfile);
laterWrong.learning={
  quizHistory:[
    {score:100,skills:['React'],perSkillScores:{React:100},questions:[{id:'react-state',skill:'React',correct:true}]},
    {score:0,skills:['React'],perSkillScores:{React:0},questions:[{id:'react-state',skill:'React',correct:false}]}
  ],
  lastQuiz:{score:0,skills:['React'],perSkillScores:{React:0},questions:[{id:'react-state',skill:'React',correct:false}]}
};
assert.equal(Core.getSkillEvidence('React',fullstack,laterWrong).assessmentEvidence.score,0,'latest matching per-skill attempt should replace older assessment evidence');

const fitBefore=Core.eligibility(Core.COMPANIES[0],fullstack,baseProfile)[2].profileFit;
const fitAfter=Core.eligibility(Core.COMPANIES[0],fullstack,jsReactCorrectNodeWrong)[2].profileFit;
assert.equal(Core.eligibility(Core.COMPANIES[0],fullstack,baseProfile)[0],Core.eligibility(Core.COMPANIES[0],fullstack,jsReactCorrectNodeWrong)[0],'Weekly Check evidence must not change deterministic company eligibility');
assert(fitAfter>=fitBefore,'Weekly Check evidence may affect company profile fit through role compatibility');

console.log('weekly skill check tests passed');
