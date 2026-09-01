const http=require('http'),fs=require('fs'),path=require('path'),crypto=require('crypto'),https=require('https'),{PDFParse}=require('pdf-parse'),mammoth=require('mammoth'),intelligence=require('./intelligence');
const loadEnv=file=>{try{for(let line of fs.readFileSync(file,'utf8').split(/\r?\n/)){line=line.trim();if(!line||line.startsWith('#')||!line.includes('='))continue;let[k,v]=line.split(/=(.*)/s);k=k.trim();if(!process.env[k])process.env[k]=String(v||'').trim().replace(/^['"]|['"]$/g,'')}}catch{}};
loadEnv(path.join(__dirname,'.env'));
loadEnv(path.join(__dirname,'backend','.env'));
const DB=process.env.ASCENTRA_DB_PATH||path.join(__dirname,'data.json'),sessions=new Map(),types={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8'};
const read=()=>{try{return JSON.parse(fs.readFileSync(DB,'utf8'))}catch{return {users:[]}}},write=d=>fs.writeFileSync(DB,JSON.stringify(d,null,2));
const hash=p=>{let s=crypto.randomBytes(16).toString('hex');return s+':'+crypto.scryptSync(p,s,64).toString('hex')},verify=(p,h)=>{let[s,v]=h.split(':');return crypto.timingSafeEqual(Buffer.from(v,'hex'),crypto.scryptSync(p,s,64))};
const cookies=r=>Object.fromEntries((r.headers.cookie||'').split(';').map(x=>x.trim().split('=').map(decodeURIComponent)).filter(x=>x[0])),user=r=>{let id=sessions.get(cookies(r).ascentra);return id&&read().users.find(x=>x.id===id)};
const persistUser=(db,me)=>{let i=db.users.findIndex(x=>x.id===me.id);if(i>=0){db.users[i]=me;write(db)}};
let requestOrigin='*';const json=(r,s,d,h={})=>{if(r.destroyed||r.headersSent||r.writableEnded)return false;r.writeHead(s,{'Content-Type':'application/json','Access-Control-Allow-Origin':r._ascentraOrigin||requestOrigin,'Access-Control-Allow-Credentials':'true',...h});r.end(JSON.stringify(d));return true},body=r=>new Promise((yes,no)=>{let x='';r.on('data',c=>x+=c);r.on('end',()=>{try{yes(JSON.parse(x||'{}'))}catch{no()}});r.on('error',no);r.on('aborted',()=>no(Object.assign(new Error('Request aborted'),{code:'CLIENT_ABORTED'})))});
const gemini=payload=>new Promise((resolve,reject)=>{let key=process.env.GEMINI_API_KEY;if(!key)return reject(new Error('ASCENTRA AI is not configured.'));let data=JSON.stringify(payload),request=https.request({hostname:'generativelanguage.googleapis.com',path:`/v1beta/models/${process.env.GEMINI_MODEL||'gemini-2.5-flash'}:generateContent?key=${encodeURIComponent(key)}`,method:'POST',headers:{'Content-Type':'application/json','Content-Length':Buffer.byteLength(data)}},response=>{let raw='';response.on('data',c=>raw+=c);response.on('end',()=>{try{let parsed=JSON.parse(raw);if(response.statusCode>=300)throw new Error(parsed.error?.message||'ASCENTRA AI request failed.');resolve(parsed.candidates?.[0]?.content?.parts?.map(p=>p.text||'').join('').trim())}catch(error){reject(error)}})});request.on('error',reject);request.write(data);request.end()});
const extractResume=async(name,encoded)=>{let data=Buffer.from(encoded||'','base64'),ext=path.extname(name||'').toLowerCase();if(!data.length||data.length>5*1024*1024)throw new Error('Resume must be a non-empty file under 5 MB.');if(ext==='.docx')return (await mammoth.extractRawText({buffer:data})).value;if(ext==='.pdf'){let parser=new PDFParse({data});try{return (await parser.getText()).text}finally{await parser.destroy()}}throw new Error('Please upload a PDF or DOCX resume.');};
const moduleAUrl=()=>process.env.MODULE_A_API_URL||'http://127.0.0.1:8001';
const aiBackendUrl=()=>process.env.ASCENTRA_AI_BACKEND_URL||'http://127.0.0.1:8000';
const aiProxyTimeoutMs=()=>Number(process.env.ASCENTRA_AI_PROXY_TIMEOUT_MS||35000);
const localAiProfile=c=>{
  c=c&&typeof c==='object'?c:{};
  let student={...(c.student&&typeof c.student==='object'?c.student:{})};
  if(c.targetRole?.id)student.target=c.targetRole.id;
  if(c.skills&&typeof c.skills==='object')student.skills=c.skills;
  if(c.academicRisk)student.academicRiskResult=c.academicRisk;
  return {student,resume:c.resume&&typeof c.resume==='object'?c.resume:{},learning:c.learning&&typeof c.learning==='object'?c.learning:{}};
};
const validateAcademicRiskPayload=b=>{
 let payload={student_id:b.student_id||b.studentId||null,semester:Number(b.semester),attendance_percentage:Number(b.attendance_percentage),ca1_score:Number(b.ca1_score),ca2_score:Number(b.ca2_score),ca3_score:Number(b.ca3_score),mid_term_score:Number(b.mid_term_score),previous_semester_tgpa:b.previous_semester_tgpa===null||b.previous_semester_tgpa===''?null:Number(b.previous_semester_tgpa),academic_trend:String(b.academic_trend||'').trim(),use_gemini:!!b.use_gemini};
 if(!Number.isInteger(payload.semester)||payload.semester<1||payload.semester>8)return {error:'semester must be an integer from 1 to 8.'};
 for(let [field,min,max] of [['attendance_percentage',0,100],['ca1_score',0,30],['ca2_score',0,30],['ca3_score',0,30],['mid_term_score',0,30]])if(!Number.isFinite(payload[field])||payload[field]<min||payload[field]>max)return {error:`${field} must be between ${min} and ${max}.`};
 payload.best_2_ca_average=b.best_2_ca_average==null||b.best_2_ca_average===''?Math.round((([payload.ca1_score,payload.ca2_score,payload.ca3_score].sort((a,b)=>b-a).slice(0,2).reduce((n,x)=>n+x,0)/2))*10)/10:Number(b.best_2_ca_average);
 if(!Number.isFinite(payload.best_2_ca_average)||payload.best_2_ca_average<0||payload.best_2_ca_average>30)return {error:'best_2_ca_average must be between 0 and 30.'};
 if(payload.semester===1)payload.previous_semester_tgpa=null;else if(!Number.isFinite(payload.previous_semester_tgpa)||payload.previous_semester_tgpa<0||payload.previous_semester_tgpa>10)return {error:'previous_semester_tgpa is required for semester 2+ and must be between 0 and 10.'};
 if(!['Declining','Stable','Improving'].includes(payload.academic_trend))return {error:'academic_trend must be Declining, Stable, or Improving.'};
 return {payload};
};
const postModuleA=payload=>new Promise((resolve,reject)=>{let base;try{base=new URL('/api/academic-risk',moduleAUrl())}catch{return reject(Object.assign(new Error('Invalid MODULE_A_API_URL.'),{status:500}))}let data=JSON.stringify(payload),transport=base.protocol==='https:'?https:http,request=transport.request({hostname:base.hostname,port:base.port||undefined,path:base.pathname,method:'POST',headers:{'Content-Type':'application/json','Content-Length':Buffer.byteLength(data)},timeout:Number(process.env.MODULE_A_TIMEOUT_MS||8000)},response=>{let raw='';response.on('data',c=>raw+=c);response.on('end',()=>{let parsed;try{parsed=JSON.parse(raw||'{}')}catch{return reject(Object.assign(new Error('Invalid response from Module A service.'),{status:502}))}if(response.statusCode>=300)return reject(Object.assign(new Error(parsed.detail||parsed.error||'Module A academic-risk request failed.'),{status:response.statusCode,details:parsed}));resolve(parsed)})});request.on('timeout',()=>request.destroy(Object.assign(new Error('Module A academic-risk service timed out.'),{status:504})));request.on('error',error=>reject(Object.assign(error,{status:error.status||502})));request.write(data);request.end()});
const checkAiBackend=()=>{let base;try{base=new URL('/api/ai/career-guide/health',aiBackendUrl())}catch{return console.log('ASCENTRA AI backend: invalid ASCENTRA_AI_BACKEND_URL')};let transport=base.protocol==='https:'?https:http,req=transport.request({hostname:base.hostname,port:base.port||undefined,path:base.pathname,method:'GET',timeout:2000},r=>{let raw='';r.on('data',c=>raw+=c);r.on('end',()=>console.log(`ASCENTRA AI backend: connected ${aiBackendUrl()} status=${r.statusCode}`))});req.on('timeout',()=>req.destroy(Object.assign(new Error('AI backend health timed out'),{code:'ETIMEDOUT'})));req.on('error',e=>console.log(`ASCENTRA AI backend: unavailable at ${aiBackendUrl()} (${e.code||e.message})`));req.end()};
http.createServer(async(req,res)=>{let url=new URL(req.url,'http://x');requestOrigin=req.headers.origin||'*';res._ascentraOrigin=requestOrigin;if(req.method==='OPTIONS'){res.writeHead(204,{'Access-Control-Allow-Origin':requestOrigin,'Access-Control-Allow-Credentials':'true','Access-Control-Allow-Headers':'Content-Type','Access-Control-Allow-Methods':'GET,POST,PUT,DELETE,OPTIONS'});return res.end()}if(url.pathname.startsWith('/api/'))try{let b=['POST','PUT'].includes(req.method)?await body(req):{},db=read(),me=user(req);
 if(url.pathname==='/api/register'){let{username,email,password}=b;username=String(username||b.name||'').trim();if(!/^[A-Za-z0-9_.-]{3,32}$/.test(username)||!/^\S+@\S+\.\S+$/.test(email)||String(password).length<8)return json(res,400,{error:'Use a username (3–32 letters, numbers, . _ -), valid email, and password of 8+ characters.'});if(db.users.some(x=>x.email===email.toLowerCase()||String(x.username||'').toLowerCase()===username.toLowerCase()))return json(res,409,{error:'An account already uses this email or username.'});let id=crypto.randomUUID(),t=crypto.randomBytes(32).toString('hex');db.users.push({id,username,name:username,email:email.toLowerCase(),password:hash(password),profile:null,hasLoggedIn:true});write(db);sessions.set(t,id);return json(res,201,{user:{id,username,email},onboarding:false,firstLogin:true},{'Set-Cookie':`ascentra=${t}; HttpOnly; SameSite=Lax; Path=/${b.remember?'; Max-Age=2592000':''}`})}
  if(url.pathname==='/api/login'){let a=db.users.find(x=>x.email===String(b.email||'').toLowerCase());if(!a||!verify(String(b.password||''),a.password))return json(res,401,{error:'Email or password is incorrect.'});let firstLogin=!a.hasLoggedIn;a.hasLoggedIn=true;write(db);let t=crypto.randomBytes(32).toString('hex');sessions.set(t,a.id);return json(res,200,{user:{id:a.id,username:a.username||a.name,email:a.email},onboarding:!!a.profile?.onboarding?.complete,profile:a.profile,firstLogin},{'Set-Cookie':`ascentra=${t}; HttpOnly; SameSite=Lax; Path=/${b.remember?'; Max-Age=2592000':''}`})}
 if(url.pathname==='/api/session'){if(!me)return json(res,401,{error:'Signed out'});return json(res,200,{user:{id:me.id,username:me.username||me.name,email:me.email},onboarding:!!me.profile?.onboarding?.complete,profile:me.profile})}
 if(url.pathname==='/api/student/profile'){if(!me)return json(res,401,{error:'Signed out'});return json(res,200,{profile:me.profile||null})}
 if(url.pathname==='/api/profile'&&req.method==='GET'){if(!me)return json(res,401,{error:'Signed out'});return json(res,200,{profile:me.profile||null})}
 if(url.pathname==='/api/profile'&&req.method==='PUT'){if(!me)return json(res,401,{error:'Signed out'});me.profile=b;persistUser(db,me);return json(res,200,{ok:true})}
 if(url.pathname==='/api/profile/update'&&req.method==='POST'){if(!me)return json(res,401,{error:'Signed out'});let current=me.profile||{},student=current.student||{},allowed=['fullName','department','university','degree','semester','graduationYear','cgpa','academicStatus','skills','softSkills','projects','experience','certifications','links','interests','secondary','academicRiskInput','academicRiskResult'];for(let key of allowed)if(Object.hasOwn(b,key))student[key]=b[key];current.student=student;me.profile=current;persistUser(db,me);return json(res,200,{ok:true,profile:current})}
 if(url.pathname==='/api/ai/guide'&&req.method==='POST'){if(!me)return json(res,401,{error:'Signed out'});if(!String(b.message||'').trim())return json(res,400,{error:'A message is required.'});return json(res,200,intelligence.guide(me.profile||{},b.message))}
 if(url.pathname==='/api/roles')return json(res,200,{roles:intelligence.readRoles(),provenance:'Curated prototype knowledge base; not live labour-market data.'})
 if(url.pathname==='/api/readiness'){if(!me)return json(res,401,{error:'Signed out'});return json(res,200,intelligence.summary(me.profile||{}))}
 if(url.pathname==='/api/career/recommendations'){if(!me)return json(res,401,{error:'Signed out'});return json(res,200,{recommendations:intelligence.recommendations(me.profile||{})})}
 if(url.pathname==='/api/career/select-role'&&req.method==='POST'){if(!me)return json(res,401,{error:'Signed out'});let found=intelligence.readRoles().find(x=>x.role_id===b.role_id);if(!found)return json(res,400,{error:'Unknown role.'});me.profile=me.profile||{};me.profile.student=me.profile.student||{};me.profile.student.target=found.role_id;me.profile.student.roleSelectedAt=Date.now();persistUser(db,me);return json(res,200,{ok:true,role:found.name,readiness:intelligence.summary(me.profile)})}
 if(url.pathname==='/api/academic-risk'&&req.method==='POST'){
   if(!me)return json(res,401,{error:'Signed out'});
   let checked=validateAcademicRiskPayload(b);
   if(checked.error)return json(res,400,{error:checked.error});
   checked.payload.student_id=checked.payload.student_id||me.id;
   try{
     let result=await postModuleA(checked.payload);
     return json(res,200,{ok:true,input:checked.payload,result});
   }catch(error){
     console.error('Module A academic-risk proxy failed:',error.message);
     return json(res,error.status||502,{error:error.status===504?error.message:'Academic Risk service is unavailable. Please make sure Module A is running.',detail:error.message,retry:true});
   }
 }
  if(url.pathname==='/api/resume/extract'&&req.method==='POST'){if(!me)return json(res,401,{error:'Signed out'});let text=await extractResume(b.name,b.data);if(text.trim().length<40)return json(res,422,{error:'Little or no readable text was found in this file.'});return json(res,200,{text})}
  if(url.pathname==='/api/ai/career-guide/health'&&req.method==='GET'){
    let backend;try{backend=new URL('/api/ai/career-guide/health',aiBackendUrl())}catch{return json(res,503,{backend:'unavailable',ai_status:'unavailable',error:'AI_BACKEND_CONFIG_ERROR'})}
    let started=Date.now(),transport=backend.protocol==='https:'?https:http,healthReq=transport.request({hostname:backend.hostname,port:backend.port||undefined,path:backend.pathname,method:'GET',timeout:3000},healthRes=>{let raw='';healthRes.on('data',c=>raw+=c);healthRes.on('end',()=>{try{return json(res,healthRes.statusCode,JSON.parse(raw||'{}'))}catch{return json(res,502,{backend:'unavailable',ai_status:'unavailable',error:'AI_BACKEND_INVALID_RESPONSE'})}})});
    healthReq.on('timeout',()=>healthReq.destroy(Object.assign(new Error('AI backend health timed out'),{code:'ETIMEDOUT'})));
    healthReq.on('error',err=>{console.error(`AI health proxy failed status=${err.code||err.message} latency_ms=${Date.now()-started}`);return json(res,503,{backend:'unavailable',ai_status:'unavailable',error:err.code==='ETIMEDOUT'?'AI_BACKEND_TIMEOUT':'AI_BACKEND_UNAVAILABLE'})});
    return healthReq.end();
  }
  if(url.pathname==='/api/career-guide'||url.pathname==='/api/ai/career-guide/chat'){
    if(!me)me={id:'local-browser-profile',username:'local-browser',email:'',profile:localAiProfile(b.context)};
    if(!me)return json(res,401,{error:'Signed out'});
    let forwardedProfile=localAiProfile(b.context);
    if(me?.profile?.student){
      forwardedProfile.student={...forwardedProfile.student,id:me.id,name:forwardedProfile.student.name||me.profile.student.name,department:forwardedProfile.student.department||me.profile.student.department,semester:forwardedProfile.student.semester||me.profile.student.semester,cgpa:forwardedProfile.student.cgpa??me.profile.student.cgpa,backlogs:forwardedProfile.student.backlogs??me.profile.student.backlogs,target:forwardedProfile.student.target||me.profile.student.target};
    }
    let postData=JSON.stringify(b),requestId=crypto.randomUUID(),backend;
    try{backend=new URL('/api/ai/career-guide/chat',aiBackendUrl())}catch{return json(res,503,{ok:false,error:'AI_BACKEND_CONFIG_ERROR',retryable:false})}
    let started=Date.now();
    let settled=false,clientGone=false,upstreamStatus=null,transport=backend.protocol==='https:'?https:http;
    const finish=(status,payload,extra={})=>{
      if(settled)return false;
      settled=true;
      let sent=!clientGone&&json(res,status,payload);
      console.log(`Ascentra AI proxy request_id=${requestId} upstream_status=${upstreamStatus||'-'} final_status=${status} latency_ms=${Date.now()-started} response_sent=${sent}${extra.error?` error=${extra.error}`:''}`);
      return sent;
    };
    req.on('aborted',()=>{clientGone=true;if(!settled){settled=true;proxyReq.destroy(Object.assign(new Error('Client aborted AI proxy request.'),{code:'CLIENT_ABORTED'}));console.log(`Ascentra AI proxy request_id=${requestId} upstream_status=${upstreamStatus||'-'} final_status=499 latency_ms=${Date.now()-started} response_sent=false error=CLIENT_ABORTED`)}});
    res.on('close',()=>{if(!res.writableEnded&&!settled){clientGone=true;settled=true;proxyReq.destroy(Object.assign(new Error('Client closed AI proxy response.'),{code:'CLIENT_CLOSED'}));console.log(`Ascentra AI proxy request_id=${requestId} upstream_status=${upstreamStatus||'-'} final_status=499 latency_ms=${Date.now()-started} response_sent=false error=CLIENT_CLOSED`)}});
    let proxyReq=transport.request({
      hostname:backend.hostname,
      port:backend.port||undefined,
      path:backend.pathname,
      method:'POST',
      headers:{
        'Content-Type':'application/json',
        'Content-Length':Buffer.byteLength(postData),
        'Cookie':req.headers.cookie||'',
        'X-Request-Id':requestId,
        'X-Ascentra-User':Buffer.from(JSON.stringify({id:me?.id,username:me?.username||me?.name,email:me?.email,profile:forwardedProfile})).toString('base64')
      },
      timeout:aiProxyTimeoutMs()
    },proxyRes=>{
      let raw='';
      upstreamStatus=proxyRes.statusCode;
      proxyRes.on('data',c=>raw+=c);
      proxyRes.on('end',()=>{
        try{
          let data=JSON.parse(raw||'{}');
          return finish(proxyRes.statusCode,data);
        }catch(e){
          return finish(502,{ok:false,error:'AI_BACKEND_INVALID_RESPONSE',retryable:true},{error:'AI_BACKEND_INVALID_RESPONSE'});
        }
      });
      proxyRes.on('aborted',()=>finish(502,{ok:false,error:'AI_BACKEND_CONNECTION_RESET',retryable:true},{error:'UPSTREAM_RESPONSE_ABORTED'}));
      proxyRes.on('error',err=>finish(502,{ok:false,error:'AI_BACKEND_UNAVAILABLE',retryable:true},{error:err.code||err.message}));
    });
    proxyReq.on('timeout',()=>{
      proxyReq.destroy(Object.assign(new Error('AI backend proxy timed out.'),{code:'ETIMEDOUT'}));
    });
    proxyReq.on('error',err=>{
      console.error('Career guide Python proxy failed:',err.code||err.message);
      return finish(err.code==='ETIMEDOUT'?504:502,{ok:false,error:err.code==='ETIMEDOUT'?'AI_BACKEND_TIMEOUT':'AI_BACKEND_UNAVAILABLE',retryable:true},{error:err.code||err.message});
    });
    proxyReq.write(postData);
    return proxyReq.end();
  }
 if(url.pathname==='/api/logout'){sessions.delete(cookies(req).ascentra);return json(res,200,{ok:true},{'Set-Cookie':'ascentra=; HttpOnly; Path=/; Max-Age=0'})}
 if(url.pathname==='/api/export'){if(!me)return json(res,401,{error:'Signed out'});return json(res,200,{account:{id:me.id,name:me.name,email:me.email},profile:me.profile})}
 if(url.pathname==='/api/account'&&req.method==='DELETE'){if(!me)return json(res,401,{error:'Signed out'});db.users=db.users.filter(x=>x.id!==me.id);write(db);sessions.delete(cookies(req).ascentra);return json(res,200,{ok:true},{'Set-Cookie':'ascentra=; HttpOnly; Path=/; Max-Age=0'})}
 return json(res,404,{error:'Not found'});
 }catch{return json(res,400,{error:'Request could not be processed.'})}
 let clean=url.pathname==='/'?'index.html':url.pathname.replace(/^\//,''),file=path.join(__dirname,clean);if(!file.startsWith(__dirname))return res.writeHead(403).end();fs.readFile(file,(e,d)=>{if(e)return res.writeHead(404).end('Not found');res.writeHead(200,{'Content-Type':types[path.extname(file)]||'application/octet-stream','Cache-Control':'no-store'});res.end(d)})
}).listen(process.env.PORT||4173,()=>{console.log(`Ascentra running at http://localhost:${process.env.PORT||4173}`);checkAiBackend()}).on('error',error=>{if(error.code==='EADDRINUSE'){console.error(`Port ${process.env.PORT||4173} is already in use. Stop the existing server or run: $env:PORT=4174; node server.js`);process.exit(1)}throw error});
