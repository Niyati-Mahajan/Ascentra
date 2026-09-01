const assert=require('assert');
const fs=require('fs');
const path=require('path');

const app=fs.readFileSync(path.join(__dirname,'..','app.js'),'utf8');
const css=fs.readFileSync(path.join(__dirname,'..','styles.css'),'utf8');

assert(app.includes("localStorage.sidebarCollapsed==='true'"),'Sidebar preference should hydrate from localStorage.sidebarCollapsed');
assert(app.includes("localStorage.sidebarCollapsed=sidebarCollapsed?'true':'false'"),'Sidebar preference should persist locally only');
assert(app.includes("data-sidebar-toggle"),'Desktop sidebar should have an edge toggle');
assert(app.includes("aria-label=\"${sidebarCollapsed?'Expand navigation':'Collapse navigation'}\""),'Toggle aria-label should reflect collapsed state');
assert(app.includes("data-sidebar-menu"),'Mobile should expose a navigation menu button');
assert(app.includes("data-sidebar-close"),'Mobile drawer should have a backdrop close target');
assert(app.includes("e.key==='Escape'"),'Escape should close the mobile drawer');
assert(app.includes("class=\"nav-tooltip\""),'Collapsed nav items should include accessible tooltip text');
assert(app.includes("class=\"profile-avatar-mini\""),'Collapsed profile card should have a compact avatar');
assert(app.includes("view=b.dataset.nav||b.dataset.view"),'Existing route binding should remain data-nav/data-view based');

['Dashboard','Role explorer','Companies','Roadmap','What-if lab','Constellation','My profile','Settings','Weekly check'].forEach(label=>{
  assert(app.includes(`'${label}'`),`Sidebar should include ${label}`);
});

assert(css.includes('--sidebar-expanded: 250px'),'Expanded sidebar width should remain 250px');
assert(css.includes('--sidebar-collapsed: 78px'),'Collapsed sidebar width should be 78px');
assert(css.includes('grid-template-columns: var(--sidebar-collapsed) minmax(0, 1fr)'),'Collapsed shell should free space for main content');
assert(css.includes('.shell.sidebar-open .side'),'Mobile drawer should slide in when opened');
assert(css.includes('@media (prefers-reduced-motion: reduce)'),'Sidebar should respect reduced-motion preference');

console.log('sidebar shell tests passed');
