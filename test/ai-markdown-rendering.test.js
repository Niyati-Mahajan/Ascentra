const assert=require('assert');
const fs=require('fs');
const path=require('path');
const vm=require('vm');

const app=fs.readFileSync(path.join(__dirname,'..','app.js'),'utf8');
const start=app.indexOf('function aiPlainText');
const end=app.indexOf('function ascentraAIWidget');
assert(start>0&&end>start,'AI Markdown rendering helpers must exist before the AI widget');

const source=`function esc(s){return String(s).replace(/[&<>]/g,x=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[x]))}\n${app.slice(start,end)};globalThis.helpers={aiMarkdown,aiMessageHtml};`;
const sandbox={};
vm.runInNewContext(source,sandbox);

const {aiMarkdown,aiMessageHtml}=sandbox.helpers;
const rendered=aiMarkdown(`**DSA is important**

1. Understand the problem
2. Analyze complexity
3. Consider edge cases

\`O(n)\``);

assert(rendered.includes('<strong>DSA is important</strong>'),'Assistant Markdown should render bold text');
assert(rendered.includes('<ol>'),'Assistant Markdown should render numbered lists');
assert(rendered.includes('<li>Understand the problem</li>'),'Assistant Markdown should render ordered list items');
assert(rendered.includes('<code>O(n)</code>'),'Assistant Markdown should render inline code');
assert(!rendered.includes('**DSA is important**'),'Raw bold Markdown should not be shown');

const unsafe=aiMessageHtml({role:'assistant',content:'**Safe** <img src=x onerror=alert(1)>'});
assert(unsafe.includes('<strong>Safe</strong>'),'Assistant bubble should use rendered Markdown');
assert(unsafe.includes('&lt;img src=x onerror=alert(1)&gt;'),'Assistant output should be escaped before rendering');
assert(!unsafe.includes('<img src=x'),'Assistant output must not inject unsafe HTML');

const user=aiMessageHtml({role:'user',content:'**plain** `code`'});
assert(user.includes('**plain** `code`'),'User messages should remain plain escaped text');
assert(!user.includes('<strong>plain</strong>'),'User messages should not render Markdown');

console.log('ai markdown rendering tests passed');
