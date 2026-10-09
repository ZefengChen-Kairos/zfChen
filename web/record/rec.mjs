import { chromium } from '/opt/node-tools/node_modules/playwright/index.mjs';
import fs from 'fs';
const [,, url, script, grid, outdir, maxSec, vw, vh, extra] = process.argv;
const W=+vw||1280, H=+vh||900, EXTRA=+extra||4;
const b = await chromium.launch({args:["--use-gl=angle","--use-angle=swiftshader","--enable-unsafe-swiftshader","--ignore-gpu-blocklist"]});
const ctx = await b.newContext({viewport:{width:W,height:H}, recordVideo:{dir:outdir, size:{width:W,height:H}}});
const pg = await ctx.newPage();
const errs=[]; pg.on('pageerror', e=>errs.push('pageerror: '+e.message));
await pg.goto(url); await pg.waitForTimeout(2500);
await pg.selectOption('#grid', grid); await pg.waitForTimeout(800);
await pg.selectOption('#scriptsel', script);
await pg.evaluate(()=>{ const b=document.querySelector('button.view[data-el="80"]'); if(b) b.click(); }); await pg.waitForTimeout(300);
const box = await pg.evaluate(()=>{ const c=document.getElementById('gl'); const r=c.getBoundingClientRect(); return {x:Math.round(r.x),y:Math.round(r.y),w:Math.round(r.width),h:Math.round(r.height)}; });
const total = JSON.parse(fs.readFileSync('scripts.json','utf8'))[script].moves.reduce((a,m)=>a+m.dur,0);
const t0=Date.now(); await pg.click('#auto');
let last=-1, tEnd=-1, tStart=-1;
while(Date.now()-t0 < (+maxSec)*1000){
  await pg.waitForTimeout(400);
  const t = await pg.evaluate(()=>{ const h=document.getElementById('hud'); const m=h&&h.textContent.match(/t\s+([\d.]+)\s*s/); return m?+m[1]:-1; });
  if(t>=0 && Math.floor(t)!==last){ last=Math.floor(t); console.log('model t', t.toFixed(1), 'real', ((Date.now()-t0)/1000).toFixed(1)); }
  if(t>=0 && tStart<0){ tStart=(Date.now()-t0)/1000; }
  if(t>=total+1.5+EXTRA){ tEnd=(Date.now()-t0)/1000; break; }
}
await pg.waitForTimeout(500);
const vpath = await pg.video().path();
await ctx.close(); await b.close();
fs.writeFileSync(outdir+'/box.json', JSON.stringify({box, total, tStart, tEnd, realPerModel: tEnd>0 ? (tEnd - tStart)/(total+1.5+EXTRA) : null, webm: vpath}));
console.log('video', vpath, JSON.stringify({box,total,tStart,tEnd})); console.log(errs.slice(0,8).join('\n')||'no errors');
