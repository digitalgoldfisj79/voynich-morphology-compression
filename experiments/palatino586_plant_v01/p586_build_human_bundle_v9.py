#!/usr/bin/env python3
"""Build the complete ZIP and serve it through a short-lived token-gated tunnel."""
import requests
URL="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/3aa5ddd6e9a298ab2fb600993bcf044959dc617e/experiments/palatino586_plant_v01/p586_build_human_bundle_v4.py"
source=requests.get(URL,timeout=120).text
source='import http.server,threading,subprocess,secrets,urllib.parse\n'+source
old='''        ticket=S.post(TICKET,headers={"x-upload-token":TOKEN},json={},timeout=120);ticket.raise_for_status();info=ticket.json();url=info.get("signedUrl") or info.get("signedURL");url=url if url.startswith("http") else SUPA+url
        with open(out,"rb") as f:
            up=S.put(url,headers={"content-type":"application/zip","x-upsert":"true"},data=f,timeout=3600);up.raise_for_status()
        meta={"protocol_id":PROTOCOL,"path":info["path"],"bytes":size,"entries":entries,"sha256":digest,"zip_test":"pass","analysis_result_sha256":analysis.get("result_sha256"),"blind_result_sha256":blind.get("result_sha256"),"closeout_result_sha256":close.get("closeout_result_sha256")}
'''
new='''        meta={"protocol_id":PROTOCOL,"path":"temporary-token-gated-transfer","bytes":size,"entries":entries,"sha256":digest,"zip_test":"pass","analysis_result_sha256":analysis.get("result_sha256"),"blind_result_sha256":blind.get("result_sha256"),"closeout_result_sha256":close.get("closeout_result_sha256")}
'''
if source.count(old)!=1:raise RuntimeError(f"handoff patch point mismatch: {source.count(old)}")
source=source.replace(old,new,1)
old2='''        print("RESULT_JSON="+json.dumps(meta,sort_keys=True),flush=True)
'''
new2='''        print("RESULT_JSON="+json.dumps(meta,sort_keys=True),flush=True)
        access_token=secrets.token_urlsafe(36)
        filename=out.name
        class TokenHandler(http.server.SimpleHTTPRequestHandler):
            def _allowed(self):
                q=urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                return secrets.compare_digest(q.get("token",[""])[0],access_token)
            def do_GET(self):
                if not self._allowed():self.send_error(403);return
                self.path=urllib.parse.urlparse(self.path).path
                return super().do_GET()
            def do_HEAD(self):
                if not self._allowed():self.send_error(403);return
                self.path=urllib.parse.urlparse(self.path).path
                return super().do_HEAD()
            def log_message(self,fmt,*args):print("HTTP "+fmt%args,flush=True)
        os.chdir(td)
        server=http.server.ThreadingHTTPServer(("0.0.0.0",8000),TokenHandler)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        proc=subprocess.Popen(["cloudflared","tunnel","--url","http://127.0.0.1:8000","--no-autoupdate"],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        public=None
        deadline=time.time()+120
        while time.time()<deadline:
            line=proc.stdout.readline()
            if line:print("TUNNEL "+line.rstrip(),flush=True)
            m=re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com",line or "")
            if m:public=m.group(0);break
            if proc.poll() is not None:break
        if not public:raise RuntimeError("cloudflared tunnel URL not obtained")
        print("DOWNLOAD_URL="+public+"/"+filename+"?token="+access_token,flush=True)
        print("DOWNLOAD_SHA256="+digest,flush=True)
        while proc.poll() is None:time.sleep(30)
        raise RuntimeError("temporary tunnel exited before cancellation")
'''
if source.count(old2)!=1:raise RuntimeError(f"serve patch point mismatch: {source.count(old2)}")
source=source.replace(old2,new2,1)
exec(compile(source,URL,"exec"),{"__name__":"__main__"})
