const assert=require('assert');
const http=require('http');
const fs=require('fs');
const os=require('os');
const path=require('path');
const {spawn}=require('child_process');

const root=path.join(__dirname,'..');
const freePort=()=>new Promise((resolve,reject)=>{
  const server=http.createServer();
  server.listen(0,'127.0.0.1',()=>{const port=server.address().port;server.close(()=>resolve(port))});
  server.on('error',reject);
});
const request=(port,pathname,{method='GET',headers={},body}={})=>new Promise((resolve,reject)=>{
  const req=http.request({hostname:'localhost',port,path:pathname,method,headers},res=>{
    let raw='';
    res.on('data',chunk=>raw+=chunk);
    res.on('end',()=>resolve({status:res.statusCode,headers:res.headers,body:raw}));
  });
  req.on('error',reject);
  if(body)req.write(body);
  req.end();
});
const waitForServer=async(port,logs)=>{
  const started=Date.now();
  while(Date.now()-started<5000){
    try{return await request(port,'/api/roles')}
    catch{await new Promise(r=>setTimeout(r,50))}
  }
  throw new Error(`Node test server did not start\nstate:\n${logs.state()}\nstdout:\n${logs.stdout()}\nstderr:\n${logs.stderr()}`);
};
const startMockBackend=port=>http.createServer((req,res)=>{
  if(req.url==='/api/ai/career-guide/health'){
    res.writeHead(200,{'Content-Type':'application/json'});
    return res.end(JSON.stringify({backend:'ok',ai_status:'ok'}));
  }
  if(req.url==='/api/ai/career-guide/chat'){
    let raw='';
    req.on('data',chunk=>raw+=chunk);
    req.on('end',()=>{
      let mode='ok';
      try{mode=JSON.parse(raw||'{}').mode||'ok'}catch{}
      if(mode==='bad'){
        res.writeHead(400,{'Content-Type':'application/json'});
        return res.end(JSON.stringify({detail:'mock validation error'}));
      }
      if(mode==='rate'){
        res.writeHead(429,{'Content-Type':'application/json'});
        return res.end(JSON.stringify({detail:{ok:false,error:'RATE_LIMIT',retryable:true}}));
      }
      if(mode==='reset')return req.socket.destroy();
      if(mode==='timeout')return;
      res.writeHead(200,{'Content-Type':'application/json'});
      return res.end(JSON.stringify({answer:'mock ok'}));
    });
    return;
  }
  res.writeHead(404,{'Content-Type':'application/json'});
  res.end(JSON.stringify({error:'not found'}));
}).listen(port,'127.0.0.1');

(async()=>{
  const nodePort=await freePort();
  const backendPort=await freePort();
  const dbPath=path.join(os.tmpdir(),`ascentra-ai-proxy-${process.pid}.json`);
  fs.writeFileSync(dbPath,JSON.stringify({users:[]}));
  const backend=startMockBackend(backendPort);
  let stdout='',stderr='';
  const child=spawn(process.execPath,['server.js'],{
    cwd:root,
    env:{...process.env,PORT:String(nodePort),ASCENTRA_AI_BACKEND_URL:`http://127.0.0.1:${backendPort}`,ASCENTRA_AI_PROXY_TIMEOUT_MS:'250',ASCENTRA_DB_PATH:dbPath},
    stdio:['ignore','pipe','pipe']
  });
  let childError=null;
  child.on('error',error=>childError=error);
  child.stdout.on('data',chunk=>stdout+=chunk);
  child.stderr.on('data',chunk=>stderr+=chunk);
  try{
    await waitForServer(nodePort,{stdout:()=>stdout,stderr:()=>stderr+(childError?`\nspawn error: ${childError.message}`:''),state:()=>`pid=${child.pid} exitCode=${child.exitCode} signalCode=${child.signalCode}`});
    const register=await request(nodePort,'/api/register',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({username:`ProxyTest${process.pid}`,email:`proxy-${process.pid}@example.com`,password:'password123'})
    });
    assert.equal(register.status,201);
    const cookie=register.headers['set-cookie'][0].split(';')[0];
    const chat=mode=>request(nodePort,'/api/ai/career-guide/chat',{
      method:'POST',
      headers:{'Content-Type':'application/json',Cookie:cookie},
      body:JSON.stringify({message:'hi',mode})
    });
    const alive=async()=>{
      const roles=await request(nodePort,'/api/roles');
      assert.equal(roles.status,200);
    };

    let response=await chat('ok');
    assert.equal(response.status,200);
    assert.equal(JSON.parse(response.body).answer,'mock ok');
    await alive();

    response=await chat('bad');
    assert.equal(response.status,400);
    assert.equal(JSON.parse(response.body).detail,'mock validation error');
    await alive();

    response=await chat('rate');
    assert.equal(response.status,429);
    assert.equal(JSON.parse(response.body).detail.error,'RATE_LIMIT');
    await alive();

    response=await chat('reset');
    assert.equal(response.status,502);
    assert.equal(JSON.parse(response.body).error,'AI_BACKEND_UNAVAILABLE');
    await alive();

    response=await chat('timeout');
    assert.equal(response.status,504);
    assert.equal(JSON.parse(response.body).error,'AI_BACKEND_TIMEOUT');
    await alive();

    assert.equal(child.exitCode,null,'Node process should remain alive after proxy failures');
    assert(!stderr.includes('ERR_HTTP_HEADERS_SENT'),'Node must not throw ERR_HTTP_HEADERS_SENT');
    assert(!stdout.includes('ERR_HTTP_HEADERS_SENT'),'Node stdout must not include ERR_HTTP_HEADERS_SENT');
    const outcomes=(stdout.match(/Ascentra AI proxy request_id=/g)||[]).length;
    assert.equal(outcomes,5,'Each chat request should produce one final proxy outcome log');
    console.log('ai proxy response-once tests passed');
  }finally{
    child.kill();
    backend.close();
    try{fs.unlinkSync(dbPath)}catch{}
  }
})().catch(error=>{
  console.error(error);
  process.exit(1);
});
