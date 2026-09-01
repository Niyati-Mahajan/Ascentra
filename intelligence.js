const Core=require('./career-core');

const readRoles=()=>Core.ROLES.map(r=>({
  role_id:r.id,
  name:r.name,
  description:r.desc,
  required_skills:Object.entries(r.req).map(([skill,level])=>({skill,importance:level,minimum_level:level})),
  preferred_skills:r.pref||[],
  market_signal:{source:'ASCENTRA Core Intelligence canonical role configuration',updated_at:'2026-08-29',strength:null}
}));

function toCoreRole(role){
  if(!role)return Core.ROLES[0];
  return Core.ROLES.find(r=>r.id===(role.id||role.role_id))||Core.ROLES[0];
}

function roleFit(profile={},role){
  let r=toCoreRole(role),match=Core.roleMatch(r,profile,Core.COMPANIES);
  return {
    score:match.match_score,
    covered:match.strong_matches,
    developing:match.developing_skills,
    gaps:match.missing_skills,
    skill_evidence:match.skill_evidence
  };
}

function summary(profile={}){
  let target=profile.student?.target||profile.target_role||null;
  let role=Core.roleById(target,Core.ROLES);
  if(!role.id)return {readiness:null,confidence:null,target_role:null,campus_alignment:null,opportunity_access:null,next_action:null,message:'Let us understand your direction first.'};
  let fit=roleFit(profile,role),readinessParts=Core.readinessParts(role,profile,Core.COMPANIES),openGaps=Core.gaps(role,profile,Core.COMPANIES);
  let hasEvidence=fit.skill_evidence.some(x=>x.meaningfulEvidence.length);
  if(!hasEvidence)return {readiness:null,confidence:null,target_role:role.name,campus_alignment:null,opportunity_access:null,next_action:null,message:'Not enough relevant evidence yet. Add role-relevant skills, projects, or an assessment to build your signal.',fit};
  let next=openGaps[0]||null;
  return {
    readiness:readinessParts.total,
    confidence:hasEvidence?'medium':'low',
    target_role:role.name,
    campus_alignment:readinessParts.campusOpportunityAlignment,
    opportunity_access:null,
    next_action:next?{skill:next.name,reason:`${next.name} is currently the highest-priority evidence gap for ${role.name}.`}:null,
    message:'Calculated from ASCENTRA Core Intelligence using saved profile, evidence, and role requirements.',
    fit:{...fit,readinessParts}
  };
}

function recommendations(profile){
  return readRoles().map(role=>({role:role.name,role_id:role.role_id,...roleFit(profile,role)})).sort((a,b)=>b.score-a.score).slice(0,3);
}

function guide(profile,message){
  let text=String(message||'').toLowerCase(),target=Core.targetRole(profile,Core.ROLES),matches=recommendations(profile),fit=target.id?roleFit(profile,target):null;
  let projectIntent=/project|portfolio|build|building|built|worth|idea/.test(text),intent=/missing|gap|lack/.test(text)?'skill_gap':projectIntent?'project_guidance':/suit|which role|recommend/.test(text)?'career_direction':/resume/.test(text)?'resume_question':/ready|readiness|eligible/.test(text)?'placement_readiness':/compare|versus| vs /.test(text)?'role_comparison':'general_career_question';
  if(intent==='project_guidance'){
    if(!target.id)return {intent,answer:'That project can be useful, but I need your target role before I can judge portfolio value accurately. Choose a role or tell me the role you are aiming for, then I can map the project against its skill requirements.',actions:['Explore career directions']};
    let gaps=(fit?.gaps||[]).slice(0,3),covered=(fit?.covered||[]).slice(0,3),resumeProjects=profile.resume?.parsed?.projects||[],savedProjects=profile.student?.projects||[],hasSimilar=[...resumeProjects,...savedProjects.map(p=>p.name+' '+p.detail)].some(p=>String(p).toLowerCase().includes('interview'));
    let direct=text.includes('worth')||text.includes('useful')?'Yes - it is worth doing if you make it demonstrate target-role skills.':'It can be a strong portfolio project if it is shaped around target-role evidence.';
    let improve=gaps.length?`To make it stronger for ${target.name}, use it to close ${gaps.join(', ')}. Make those parts visible in the project description and resume evidence.`:`For ${target.name}, your core skill coverage already looks solid, so make the project show depth, testing, deployment, and clear user value.`;
    let evidence=hasSimilar?'I can see interview-preparation project evidence already saved, so the next step is improving its technical depth.':'I do not see this project in saved resume/profile evidence yet, so treat it as planned work until you add it.';
    return {intent,answer:`${direct} ${improve} ${covered.length?`Your current strongest related evidence is ${covered.join(', ')}. `:''}${evidence} This is guidance from your saved profile, not a placement probability.`,actions:['Add project evidence','Build my roadmap']}
  }
  if(intent==='skill_gap'&&target.id)return {intent,answer:`For ${target.name}, your Ready skills are ${fit.covered.join(', ')||'not enough role-relevant skills yet'}. Focus next on ${fit.gaps.slice(0,3).join(', ')||'turning your skills into project evidence'}.`,actions:['Build my roadmap','Explore matching roles']};
  if(intent==='career_direction')return {intent,answer:matches[0]?.score?`Based on ASCENTRA Core Intelligence, your current closest directions are ${matches.map(x=>`${x.role} (${x.score}% compatibility)`).join(', ')}. These are evidence matches, not placement predictions.`:'Your direction is still forming. Tell me what you enjoy building, learning, or solving so I can compare roles using your answers.',actions:['Explore career directions','Analyze my skills']};
  if(intent==='placement_readiness'){let s=summary(profile);return {intent,answer:s.readiness===null?s.message:`Your current ${s.target_role} readiness signal is ${s.readiness}/100 with ${s.confidence} confidence. ${s.next_action?.reason||s.message}`,actions:['Find my biggest skill gap']}};
  if(intent==='resume_question')return {intent,answer:profile.resume?.parsed?`Your resume contributes ${profile.resume.parsed.skills?.length||0} technical skill signals and ${profile.resume.parsed.projects?.length||0} project signals. Add only genuine evidence when you update it.`:'No validated resume evidence is saved yet. Upload your resume from Dashboard when you are ready.',actions:['Improve my resume']};
  return {intent,answer:target.id?`You are currently targeting ${target.name}. Ask about its skills, your gaps, projects, role comparisons, or readiness; I will use ASCENTRA Core Intelligence results from your saved evidence.`:'Your career profile is still forming. What kinds of things do you enjoy building or solving, and what technical areas have you enjoyed so far?',actions:['Explore career directions']}
}

module.exports={readRoles,summary,recommendations,roleFit,guide,Core};
