const assert=require('assert');
const fs=require('fs');
const path=require('path');
const vm=require('vm');

const app=fs.readFileSync(path.join(__dirname,'..','app.js'),'utf8');

assert(app.includes("const AI_TRANSIENT_ERROR_TEXT='I’m having trouble connecting right now. Your profile tools are still available, and you can retry in a moment.'"),'Known transient connection-card text must be centralized');
assert(app.includes('function aiIsKnownFallbackText(text)'),'Old saved fallback cards must be recognized for migration');
assert(app.includes("replace(/^I'm/,'I’m')"),'Migration must remove straight-apostrophe legacy fallback cards too');
assert(app.includes('function aiIsTransientErrorMessage(m)'),'Transient AI error entries must be identifiable');
assert(app.includes('function sanitizeGuideHistory(history)'),'Guide history must have a sanitizer');
assert(app.includes("['user','assistant'].includes(m?.role)"),'Only user and assistant messages should be persisted');
assert(app.includes('function sanitizeConversationState(data={})'),'Persisted state must be sanitized before storage or profile sync');
assert(app.includes('data=sanitizeConversationState(saved&&typeof saved'),'Hydration must migrate saved old transient error cards');
assert(app.includes('guideHistory:sanitizeGuideHistory(data.guideHistory)'),'Hydrated state must contain only persistable guide history');
assert(app.includes('let persistable=sanitizeConversationState(state);Store.save(persistable);'),'Local save must not store transient AI errors');
assert(app.includes('body:JSON.stringify(persistable)'),'/api/profile sync must not store transient AI errors');
assert(app.includes("type:'error',transient:true,content:AI_TRANSIENT_ERROR_TEXT"),'New connection failures must be marked as transient error messages');
assert(app.includes('findLastIndex(m=>m.error&&m.retry&&m.failedPrompt===q)'),'Retry must locate the current transient error card');
assert(app.includes('if(errorIndex>=0)state.guideHistory.splice(errorIndex,1)'),'Retry must remove the transient error card before retrying');
assert(app.includes('if(!retrying)state.guideHistory.push({role:\'user\',content:q,turnId:requestId})'),'Retry must not duplicate the user message');
assert(app.includes('if(state.settings.rememberConversations!==false)save()'),'Conversation memory setting must still control persistence');

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
vm.runInNewContext(`${helperSource};globalThis.helpers={hydrateState,sanitizeGuideHistory,sanitizeConversationState,Store,AI_TRANSIENT_ERROR_TEXT};`,sandbox);
const {hydrateState,sanitizeGuideHistory,sanitizeConversationState,Store,AI_TRANSIENT_ERROR_TEXT}=sandbox.helpers;
const oldCurly={role:'assistant',content:AI_TRANSIENT_ERROR_TEXT,error:true,retry:true,failedPrompt:'hi'};
const oldStraight={role:'assistant',content:AI_TRANSIENT_ERROR_TEXT.replace('I’m',"I'm"),error:true,retry:true,failedPrompt:'hi'};
const realAssistant={role:'assistant',content:'A real Gemini answer'};
const user={role:'user',content:'hi'};
assert.deepEqual(sanitizeGuideHistory([user,oldCurly,realAssistant,oldStraight]),[user,realAssistant],'Sanitizer must remove only known transient fallback cards');
assert.deepEqual(hydrateState({guideHistory:[oldCurly,user,realAssistant]}).guideHistory,[user,realAssistant],'Hydration must migrate old saved transient errors');
assert.equal(hydrateState({}).guideHistory,undefined,'Hydration must not create chat history for fresh accounts');
assert.deepEqual(sanitizeConversationState({guideHistory:[oldCurly,user]}).guideHistory,[user],'Persisted state must exclude transient errors');
Store.save({settings:{},guideHistory:[oldCurly,user,realAssistant]},'chat-test');
assert.deepEqual(JSON.parse(storage['ascentra:chat-test']).guideHistory,[oldCurly,user,realAssistant],'Store remains a raw storage adapter; save() owns sanitization');

console.log('ai chat state tests passed');
