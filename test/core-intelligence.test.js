const assert=require('assert');
const Core=require('../career-core');

const profile={
  student:{
    target:'fullstack',
    department:'CSE-AI & ML',
    cgpa:9.2,
    backlogs:0,
    skills:{JavaScript:28,React:42,'Node.js':62,'Express.js':69,SQL:61,'REST APIs':69},
    projects:[{name:'Roadmap action',detail:'deploy'}],
    internships:0,
    certifications:[],
    roadmap:{}
  },
  resume:{parsed:{text:'Projects JavaScript SQL Git GitHub',skills:['JavaScript','SQL','Git'],projects:[],sections:{projects:true}}},
  learning:{quizHistory:[{score:67,correct:2,total:3,skills:['REST APIs','React','DSA']}],lastQuiz:{score:67,correct:2,total:3,skills:['REST APIs','React','DSA']}}
};

const fullstack=Core.roleById('fullstack');
assert.equal(Core.roleScore(fullstack,profile),84,'canonical Full-Stack role score should match baseline');
assert.equal(Core.readiness(fullstack,profile,Core.COMPANIES),71,'readiness should change only because department normalization unlocks broad CSE companies');

const expected=[
  ['JavaScript',32,'Priority gap'],
  ['React',47,'Developing'],
  ['Node.js',62,'Developing'],
  ['Express.js',69,'Ready'],
  ['SQL',61,'Ready'],
  ['REST APIs',69,'Ready'],
  ['Git',45,'Developing'],
  ['DSA',55,'Developing']
];
for(const [skill,score,status] of expected){
  const ev=Core.getSkillEvidence(skill,fullstack,profile);
  assert.equal(ev.evidenceScore,score,`${skill} evidence score`);
  assert.equal(ev.status,status,`${skill} status`);
}

assert(Core.departmentPass('CSE-AI & ML',['CSE']),'CSE specialization should pass broad CSE rule');
assert(!Core.departmentPass('ECE',['CSE']),'ECE must not pass CSE-only rule');

const northstar=Core.COMPANIES.find(c=>c.name==='Northstar Systems');
const eligible=Core.eligibility(northstar,fullstack,profile);
assert.equal(eligible[0],'ELIGIBLE','company eligibility must use stored application rules, not role skill gaps');
assert(!eligible[2].failed.includes('department'),'normalized CSE specialization should not fail the department hard rule');
assert.equal(eligible[2].profileFit,Core.roleScore(fullstack,profile),'company profile fit must use canonical role score');
assert(eligible[2].preparationGaps.includes('JavaScript'),'skill gaps remain available as preparation signals');

const lowSkillProfile=structuredClone(profile);
lowSkillProfile.student.skills.JavaScript=0;
lowSkillProfile.resume={parsed:{text:'',skills:[],projects:[],sections:{}}};
assert.equal(Core.eligibility(northstar,fullstack,lowSkillProfile)[0],'ELIGIBLE','low JavaScript evidence must not block eligibility without a company-specific skill cutoff');
assert(Core.roleScore(fullstack,lowSkillProfile)<Core.roleScore(fullstack,profile),'low JavaScript should still reduce profile fit');
assert(Core.gaps(fullstack,lowSkillProfile,Core.COMPANIES).some(g=>g.name==='JavaScript'),'low JavaScript should remain a preparation gap');

const lowCgpaProfile=structuredClone(profile);
lowCgpaProfile.student.cgpa=6.0;
assert.equal(Core.eligibility(northstar,fullstack,lowCgpaProfile)[0],'NOT ELIGIBLE','large CGPA gap should fail application eligibility');

const nearCgpaProfile=structuredClone(profile);
nearCgpaProfile.student.cgpa=6.8;
assert.equal(Core.eligibility(northstar,fullstack,nearCgpaProfile)[0],'NEAR ELIGIBILITY','small CGPA gap should remain near eligibility');

const wrongDeptProfile=structuredClone(profile);
wrongDeptProfile.student.department='ME';
assert.equal(Core.eligibility(northstar,fullstack,wrongDeptProfile)[0],'NOT ELIGIBLE','unsupported department should fail application eligibility');

const mlRole=Core.roleById('ml');
assert.equal(Core.eligibility(northstar,mlRole,profile)[0],'NOT ELIGIBLE','unsupported target role should fail application eligibility');

const roadmap=Core.roadmap(fullstack,profile,Core.COMPANIES).flatMap(p=>p.items).map(x=>x.skill);
assert.deepEqual(roadmap.slice(0,5),['JavaScript','React','Node.js','Git','DSA'],'roadmap must consume canonical gaps');

const roadmapBefore=structuredClone(profile);
const roadmapCompleted=structuredClone(profile);
roadmapCompleted.student.roadmap={'fullstack-JavaScript':true,'fullstack-React':true,'fullstack-Node.js':true,'fullstack-Git':true,'fullstack-DSA':true};
assert.equal(Core.getSkillEvidence('JavaScript',fullstack,roadmapCompleted).evidenceScore,Core.getSkillEvidence('JavaScript',fullstack,roadmapBefore).evidenceScore,'roadmap completion must not change skill evidence');
assert.equal(Core.getSkillEvidence('Git',fullstack,roadmapCompleted).evidenceScore,Core.getSkillEvidence('Git',fullstack,roadmapBefore).evidenceScore,'completed Git roadmap action must not validate Git evidence');
assert.equal(Core.getSkillEvidence('DSA',fullstack,roadmapCompleted).evidenceConfidence,Core.getSkillEvidence('DSA',fullstack,roadmapBefore).evidenceConfidence,'roadmap completion must not change evidence confidence');
assert.equal(Core.roleScore(fullstack,roadmapCompleted),Core.roleScore(fullstack,roadmapBefore),'roadmap completion must not change role compatibility');
assert.equal(Core.readiness(fullstack,roadmapCompleted,Core.COMPANIES),Core.readiness(fullstack,roadmapBefore,Core.COMPANIES),'roadmap completion must not change readiness');
assert.equal(Core.eligibility(northstar,fullstack,roadmapCompleted)[2].profileFit,Core.eligibility(northstar,fullstack,roadmapBefore)[2].profileFit,'roadmap completion must not change company profile fit');
assert(Core.roadmap(fullstack,roadmapCompleted,Core.COMPANIES).flatMap(p=>p.items).find(x=>x.id==='fullstack-JavaScript').done,'roadmap completion state must still persist into generated roadmap items');

const personalOnly=structuredClone(profile);
personalOnly.student.customRoadmapActions=[{id:'custom-1',text:'Build a JavaScript React Node.js API project',completed:true,createdAt:1,roleId:'fullstack'}];
assert.equal(personalOnly.student.projects.length,profile.student.projects.length,'personal roadmap actions must be stored separately from projects');
assert.equal(Core.getSkillEvidence('JavaScript',fullstack,personalOnly).evidenceScore,Core.getSkillEvidence('JavaScript',fullstack,profile).evidenceScore,'personal roadmap actions must not create skill evidence');
assert.equal(Core.roleScore(fullstack,personalOnly),Core.roleScore(fullstack,profile),'personal roadmap actions must not change role compatibility');

const withRealProject=structuredClone(profile);
withRealProject.student.projects.push({name:'Full-stack API',detail:'Implemented JavaScript frontend and REST API backend with saved data.'});
assert(Core.getSkillEvidence('JavaScript',fullstack,withRealProject).evidenceScore>Core.getSkillEvidence('JavaScript',fullstack,profile).evidenceScore,'real project evidence should still change skill evidence');
assert(Core.gaps(fullstack,withRealProject,Core.COMPANIES).find(g=>g.name==='JavaScript').gap<Core.gaps(fullstack,profile,Core.COMPANIES).find(g=>g.name==='JavaScript').gap,'roadmap should regenerate when accepted evidence changes the canonical gaps');

const simulated=structuredClone(profile);
simulated.student.skills.JavaScript=75;
assert.equal(Core.gaps(fullstack,simulated,Core.COMPANIES)[0].name,'React','what-if style simulated profile must reuse canonical gaps');
assert(Core.roleScore(fullstack,simulated)>Core.roleScore(fullstack,profile),'simulated improvement should raise canonical role score');

const whatIfBase=structuredClone(profile);
const whatIfScenario=structuredClone(whatIfBase);
whatIfScenario.student.skills.JavaScript=60;
assert.equal(whatIfBase.student.skills.JavaScript,28,'running a simulation-style clone must not mutate saved student skills');
assert.deepEqual(whatIfBase.resume,profile.resume,'simulation-style clone must not mutate accepted resume evidence');
assert.deepEqual(whatIfBase.student.projects,profile.student.projects,'simulation-style clone must not mutate accepted project evidence');
assert.equal(Core.eligibility(northstar,fullstack,whatIfBase)[0],Core.eligibility(northstar,fullstack,whatIfScenario)[0],'skill-only simulation must not change deterministic company eligibility');
assert.equal(Core.getSkillEvidence('JavaScript',fullstack,whatIfScenario).evidenceScore,60,'simulation should recalculate canonical skill evidence from hypothetical profile evidence');
assert.equal(Core.roleScore(fullstack,whatIfScenario),89,'simulation should recalculate canonical role compatibility');
assert.equal(Core.readiness(fullstack,whatIfScenario,Core.COMPANIES),73,'simulation should recalculate canonical readiness');
assert.equal(Core.gaps(fullstack,whatIfScenario,Core.COMPANIES)[0].name,'React','simulation should recalculate canonical skill gaps');
const withLastSimulation=structuredClone(profile);
withLastSimulation.learning.lastSimulation={skill:'JavaScript',to:100};
assert.equal(Core.roleScore(fullstack,withLastSimulation),Core.roleScore(fullstack,profile),'lastSimulation metadata must not affect role compatibility');
assert.equal(Core.readiness(fullstack,withLastSimulation,Core.COMPANIES),Core.readiness(fullstack,profile,Core.COMPANIES),'lastSimulation metadata must not affect readiness');

const appSource=require('fs').readFileSync(require('path').join(__dirname,'..','app.js'),'utf8');
assert(!appSource.includes("state.student.projects.push({name:'Roadmap action'"),'personal roadmap actions must not be pushed into projects');
assert(!appSource.includes('id="apply-sim"'),'What-if Lab must not render an apply-scenario action');
assert(!appSource.includes('state.student=copy;record()'),'What-if Lab must not persist simulated skill values');
assert(!appSource.includes('state.student=copy'),'What-if Lab should calculate simulated results without swapping global student state');
assert(!appSource.includes('state.learning.lastSimulation'),'What-if simulation history should not be written to persistent learning state');
assert(appSource.includes('Simulate skill evidence'),'What-if Lab should use evidence terminology');
assert(/hypothetical profile evidence/i.test(appSource),'What-if sliders should label the simulated value accurately');
assert(appSource.includes('Role requirement'),'What-if sliders should show the selected role requirement');
assert(appSource.includes('id="run-sim" disabled'),'Run simulation should start disabled until a scenario changes');
assert(appSource.includes('id="reset-sim" disabled'),'Reset simulation should start disabled until a scenario changes or result exists');
assert(!/readiness\s*[+]\s*=|roleScore\s*[+]\s*=|skillIncrease\s*[*]\s*0\.2/.test(appSource),'What-if Lab must not use simulator-specific hardcoded boosts');

console.log('core intelligence tests passed');
