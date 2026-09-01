const assert=require('assert');
const fs=require('fs');
const path=require('path');
const Core=require('../career-core');

const root=path.join(__dirname,'..');
const app=fs.readFileSync(path.join(root,'app.js'),'utf8');
const css=fs.readFileSync(path.join(root,'styles.css'),'utf8');
const server=fs.readFileSync(path.join(root,'server.js'),'utf8');

assert(app.includes('function resetBrowserStateForLogout()'),'logout must clear account-scoped browser state');
assert(app.includes('state=hydrateAccountState(null,null,{fresh:true})'),'logout reset should replace in-memory state with a fresh hydrated state');
assert(app.includes('document.querySelector(\'#logout\').onclick=logoutUser'),'Settings logout must use the privacy-safe logout path');

assert(app.includes('function clearConversationHistory()'),'Delete conversations must use a dedicated clearing helper');
assert(app.includes('state.guideHistory=[]'),'Delete conversations must clear main ASCENTRA AI guide history');
assert(app.includes('profileGuide)state.profileGuide={...state.profileGuide,messages:[]}'),'Delete conversations must clear profile guide messages');
assert(app.includes('onboarding)state.onboarding={...state.onboarding,messages:[]}'),'Delete conversations must clear onboarding messages');

assert(app.includes('Sync my career profile to my account'),'profile memory label must describe account sync');
assert(app.includes('Your profile remains available on this browser'),'profile sync helper text must explain local browser persistence');
assert(app.includes('Remember ASCENTRA AI conversations'),'AI memory label should remain visible');
assert(app.includes('prevents conversation history from being saved'),'AI memory helper text must explain persistence behavior');

assert(app.includes('context:aiRelevantContext(q)'),'Gemini requests must still receive live canonical profile context');
assert(app.includes('if(state.settings.rememberConversations!==false)save()'),'conversation memory off must affect persistence, not live context');
assert(/rememberProfile!==false\)fetch\('\/api\/profile'/.test(app),'profile sync off must prevent future /api/profile PUT calls');

assert(app.includes('View current student record'),'stored-profile action must be renamed to local student record');
assert(app.includes('function safeStudentRecord()'),'student record view must redact sensitive/noisy fields');
assert(app.includes('redacted profile photo data'),'profile photo data must be redacted from the student-record modal');
assert(app.includes('Export may include profile information, resume text, academic results'),'download disclosure must describe exported data categories');

assert(app.includes('Delete account and synced data?'),'delete account must use a stronger confirmation modal');
assert(app.includes('External service data, if any, may not be covered'),'delete account copy must avoid claiming universal deletion');
assert(app.includes("localStorage.removeItem('ascentra')"),'delete account must clear local ASCENTRA state');
assert(server.includes("account:{id:me.id,name:me.name,email:me.email},profile:me.profile"),'export must exclude password hash and session token');
assert(!/password/.test(server.match(/url\.pathname==='\/api\/export'[\s\S]*?\}/)?.[0]||''),'export route must not include password data');

assert(app.includes('setupSystemThemeListener'),'System theme should register a preference-change listener');
assert(app.includes("mq.addEventListener?.('change'"),'System theme should respond to OS preference changes');
assert(app.includes("document.body.style.setProperty('--lime',accent)"),'accent color must apply even when dark mode defines body-level --lime');
assert(css.includes('body.compact .card'),'density compact mode must have visible CSS effects');
assert(css.includes('body.no-motion *'),'motion setting must disable nonessential CSS animations/transitions');

const profile={
  student:{target:'fullstack',department:'CSE-AI & ML',cgpa:9.2,backlogs:0,skills:{JavaScript:28,React:42,'Node.js':62,'Express.js':69,SQL:61,'REST APIs':69},projects:[],internships:0,certifications:[],roadmap:{}},
  resume:{parsed:{text:'Projects JavaScript SQL Git GitHub',skills:['JavaScript','SQL','Git'],projects:[],sections:{projects:true}}},
  learning:{quizHistory:[{score:67,skills:['REST APIs','React','DSA']}],lastQuiz:{score:67,skills:['REST APIs','React','DSA']}},
  settings:{theme:'light',accent:'#c6f260',font:100,motion:true,contrast:100,density:'comfortable'}
};
const changedSettings=structuredClone(profile);
changedSettings.settings={theme:'dark',accent:'#ffcc00',font:125,motion:false,contrast:130,density:'compact',rememberProfile:false,rememberConversations:false};
const role=Core.roleById('fullstack');
assert.equal(Core.roleScore(role,changedSettings),Core.roleScore(role,profile),'Settings must not affect role compatibility');
assert.equal(Core.readiness(role,changedSettings,Core.COMPANIES),Core.readiness(role,profile,Core.COMPANIES),'Settings must not affect readiness');
assert.equal(Core.eligibility(Core.COMPANIES[0],role,changedSettings)[0],Core.eligibility(Core.COMPANIES[0],role,profile)[0],'Settings must not affect company eligibility');
assert.equal(Core.getSkillEvidence('JavaScript',role,changedSettings).evidenceScore,Core.getSkillEvidence('JavaScript',role,profile).evidenceScore,'Settings must not affect skill evidence');

console.log('settings semantics tests passed');
