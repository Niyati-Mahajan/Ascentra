const assert=require('assert');
const fs=require('fs');
const path=require('path');
const vm=require('vm');

const app=fs.readFileSync(path.join(__dirname,'..','app.js'),'utf8');

assert(app.includes("key(id){return `ascentra:${id}`}"),'account-scoped browser cache must be namespaced by user id');
assert(app.includes('localStorage.ascentraDeviceSettings'),'device preferences must be stored separately from account data');
assert(app.includes("localStorage.removeItem('ascentra')"),'legacy global full-profile storage must be cleared');
assert(!app.includes('existingResume=state?.resume'),'auth must not preserve the previous account resume before hydration');
assert(!app.includes('state.resume=existingResume'),'auth must not copy a previous account resume into the current account');
assert(app.includes('state=hydrateAccountState(d.profile,account,{fresh:register})'),'login/register must hydrate through account-isolated state');
assert(app.includes('fresh?{}:(profile||Store.get(user?.id)||{})'),'new accounts must start from clean defaults, while existing accounts may use only their own namespace');
assert(app.includes('compare=[];lastSimulation=null;window._ascentraLastPrompt=\'\''),'auth account switches must clear transient comparison, simulation, and AI retry state');
assert(app.includes('Store.saveDeviceSettings(state.settings);state=hydrateAccountState(null,null,{fresh:true})'),'logout must retain only device preferences and clear account-scoped memory');
assert(app.includes("if(id)localStorage.removeItem(Store.key(id))"),'delete account must remove only the deleted account namespace');

const submitAuth=app.match(/async function submitAuth\(e\)[\s\S]*?catch\(error\)\{err\.textContent=`Unable to connect to Ascentra\.[\s\S]*?\}\}/)?.[0]||'';
assert(submitAuth.includes('fresh:register'),'register path must request clean-account hydration');
assert(!/existingResume|state\.resume\s*=/.test(submitAuth),'submitAuth must not assign resume from old browser state');

const aiLiveContext=app.match(/function aiLiveContext\(\)[\s\S]*?academicRisk:state\.student\.academicRiskResult\|\|null\}\}/)?.[0]||'';
assert(aiLiveContext.includes('resume:state.resume||{}'),'AI live context must read resume only from current hydrated account state');
assert(aiLiveContext.includes('learning:state.learning||{}'),'AI live context must read weekly checks only from current hydrated account state');
assert(aiLiveContext.includes('lastSimulation'),'AI live context must expose only current transient simulation state');

const aiRelevantContext=app.match(/function aiRelevantContext\(q\)[\s\S]*?return trimAIContext\(picked\)\}/)?.[0]||'';
assert(aiRelevantContext.includes('let c=aiLiveContext()'),'AI relevant context must derive from current account live context');
assert(!/localStorage\.ascentra(?!DeviceSettings)/.test(aiRelevantContext),'AI context must not read legacy global localStorage state');

const helperSource=app.slice(0,app.indexOf('let state=hydrateState'));
const storage={};
const sandbox={
  structuredClone,
  localStorage:{
    get ascentra(){return storage.ascentra},
    set ascentra(value){storage.ascentra=value},
    get ascentraDeviceSettings(){return storage.ascentraDeviceSettings},
    set ascentraDeviceSettings(value){storage.ascentraDeviceSettings=value},
    getItem(key){return Object.hasOwn(storage,key)?storage[key]:null},
    setItem(key,value){storage[key]=String(value)},
    removeItem(key){delete storage[key]}
  },
  AscentraCore:{SKILLS:[],ROLES:[],COMPANIES:[]}
};
vm.runInNewContext(`${helperSource};globalThis.helpers={Store,hydrateState,hydrateAccountState};`,sandbox);
const {Store,hydrateAccountState}=sandbox.helpers;

storage.ascentra=JSON.stringify({
  settings:{theme:'dark',font:125,motion:false,density:'compact',accent:'#ffcc00',contrast:130},
  student:{name:'Account A',target:'fullstack',skills:{React:90},projects:[{name:'A project',detail:'private'}]},
  resume:{name:'account-a.pdf',text:'PRIVATE A RESUME',skills:['React'],parsed:{text:'PRIVATE A RESUME',skills:['React'],projects:['A private project'],sections:{projects:true}},versions:[{name:'account-a.pdf'}]},
  learning:{lastQuiz:{score:99,skills:['React']}},
  guideHistory:[{role:'user',content:'A private chat'}]
});

let freshB=hydrateAccountState(null,{id:'b',username:'AccountB'},{fresh:true});
assert.equal(freshB.resume.parsed,undefined,'new Account B must not inherit parsed resume');
assert.equal(freshB.resume.text,'','new Account B must not inherit raw resume text');
assert.deepEqual(freshB.resume.skills,[],'new Account B must not inherit parsed resume skills');
assert.deepEqual(freshB.student.skills,{},'new Account B must not inherit Account A skills');
assert.deepEqual(freshB.student.projects,[],'new Account B must not inherit Account A projects');
assert.equal(freshB.student.target,null,'new Account B must not inherit Account A target role');
assert.equal(freshB.learning.lastQuiz,null,'new Account B must not inherit weekly check data');
assert.equal(freshB.guideHistory,undefined,'new Account B must not inherit ASCENTRA AI history');
assert.equal(freshB.settings.theme,'dark','device theme preference should survive account switch');
assert.equal(freshB.settings.font,125,'device font preference should survive account switch');

Store.save({
  student:{name:'Account A',skills:{SQL:80},projects:[{name:'A project',detail:'private'}]},
  settings:{theme:'light',font:100,motion:true,density:'comfortable',accent:'#c6f260',contrast:100},
  resume:{name:'account-a.pdf',text:'PRIVATE A RESUME',skills:['SQL'],parsed:{text:'PRIVATE A RESUME',skills:['SQL'],projects:[],sections:{}}},
  learning:{lastQuiz:{score:88,skills:['SQL']}},
  guideHistory:[{role:'user',content:'A secret'}]
},'a');
let loginB=hydrateAccountState(null,{id:'b',username:'AccountB'},{fresh:false});
assert.equal(loginB.resume.parsed,undefined,'Account B with no server or namespaced resume must stay empty');
assert.deepEqual(loginB.student.projects,[],'Account B with no namespace must not receive Account A projects');
assert.equal(loginB.guideHistory,undefined,'Account B with no namespace must not receive Account A chat history');

let loginA=hydrateAccountState(null,{id:'a',username:'AccountA'},{fresh:false});
assert.equal(loginA.resume.name,'account-a.pdf','Account A should recover its own namespaced resume');
assert.equal(loginA.resume.text,'PRIVATE A RESUME','Account A should recover its own raw resume text');
assert.deepEqual(loginA.student.projects,[{name:'A project',detail:'private'}],'Account A should recover its own projects');
assert.deepEqual(loginA.guideHistory,[{role:'user',content:'A secret'}],'Account A should recover its own chat history');

let serverB=hydrateAccountState({student:{name:'Account B'},settings:{theme:'light'},resume:{name:'No resume uploaded',text:'',skills:[],lastUploaded:null,stats:{}}},{id:'b',username:'AccountB'},{fresh:false});
assert.equal(serverB.resume.parsed,undefined,'server profile without resume must not fall back to previous browser resume');
assert.equal(serverB.resume.text,'','server profile without resume must keep empty resume text');

console.log('account isolation tests passed');
