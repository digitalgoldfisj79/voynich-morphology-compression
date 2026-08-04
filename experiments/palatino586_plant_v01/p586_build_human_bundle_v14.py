#!/usr/bin/env python3
"""Correct v12 generated script construction without nested escape sequences."""
import requests
URL="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/4dd52fc16175102b2fc922655703ba28dc6b782f/experiments/palatino586_plant_v01/p586_build_human_bundle_v12.py"
wrapper=requests.get(URL,timeout=120).text
start=wrapper.index('        urls="\\n".join')
end=wrapper.index('        assets=[]',start)
replacement='''        nl=chr(10);dq=chr(34);sq=chr(39)
        urls=nl.join(["  "+sq+p["url"]+sq+"," for p in parts]).rstrip(",")
        ps_lines=[
            '$ErrorActionPreference = "Stop"',
            '$Output = Join-Path $PWD "P586_PLANT_MORPHOLOGY_COMPLETE.zip"',
            '$Urls = @(',urls,')',
            '$stream = [System.IO.File]::Create($Output)','try {',
            '  for ($i=0; $i -lt $Urls.Count; $i++) {',
            '    $tmp = Join-Path $env:TEMP ("p586_part_{0:D3}" -f $i)',
            '    Write-Host ("Downloading part {0}/{1}" -f ($i+1), $Urls.Count)',
            '    Invoke-WebRequest -Uri $Urls[$i] -OutFile $tmp',
            '    $bytes = [System.IO.File]::ReadAllBytes($tmp)',
            '    $stream.Write($bytes, 0, $bytes.Length)',
            '    Remove-Item $tmp -Force','  }','} finally { $stream.Dispose() }',
            '$hash=(Get-FileHash -Algorithm SHA256 $Output).Hash.ToLower()',
            'Write-Host "SHA-256: $hash"',
            f'if ($hash -ne "{digest}") {{ throw "SHA-256 mismatch" }}',
            'Write-Host "Archive assembled and verified: $Output"',
        ]
        ps=nl.join(ps_lines)+nl
        sh_lines=['#!/usr/bin/env bash','set -euo pipefail','out="P586_PLANT_MORPHOLOGY_COMPLETE.zip"',': > "$out"']
        sh_lines.extend(['curl -fL '+sq+p['url']+sq+' >> "$out"' for p in parts])
        sh_lines.extend([
            'actual=$(sha256sum '+dq+'$out'+dq+' | awk '+sq+'{print $1}'+sq+')',
            'expected='+dq+digest+dq,
            'echo '+dq+'SHA-256: $actual'+dq,
            '[ "$actual" = "$expected" ] || { echo '+sq+'SHA-256 mismatch'+sq+' >&2; exit 1; }',
            'echo "Archive assembled and verified: $out"'
        ])
        sh=nl.join(sh_lines)+nl
        readme=nl.join([
            '# P586 Plant-Morphology Evidence Bundle','',
            'The compact core ZIP contains all protocols, amendments, reports, manifests, CSV ledgers and blind evidence sheets. The full archive additionally contains every retained ordinary, masked, above-ground and reproductive crop.','',
            f'Full archive SHA-256: `{digest}`  ',f'Full archive bytes: `{size}`  ',f'Full archive entries: `{entries}`  ',f'Parts: `{len(parts)}` in ascending `part000`, `part001`, ... order.','',
            'Use `ASSEMBLE_P586_ARCHIVE.ps1` on Windows or `assemble_p586_archive.sh` on Linux/macOS. Both scripts download, concatenate and verify the archive.',''
        ])
'''
wrapper=wrapper[:start]+replacement+wrapper[end:]
exec(compile(wrapper,URL,"exec"),{"__name__":"__main__"})
