"""Build Nifty 500 point-in-time membership from niftyindices monthly zips (indices_data<Mon><YYYY>.zip,
files nx_<Mon><YYYY>.zip in cwd). Snapshots Oct2020..Mar2022; later dates frozen at Mar2022 (causal, stale).
Renames/glued symbols were then mapped by hand; see artifacts/universe/nifty500_members.csv."""
import zipfile, io, re, sys, pypdf, pandas as pd
snaps = {"Oct2020": "2020-10-30", "Apr2021": "2021-04-30", "Oct2021": "2021-10-29", "Mar2022": "2022-03-31"}
sym = re.compile(r"^([A-Z0-9&\-]{2,}?)(?=\s|[A-Z][a-z])"); num = re.compile(r"\d+\.\d{2}\s+\d+\.\d{2}\s+\d+\s*$")
lists = {}
for m in snaps:
    z = zipfile.ZipFile(f"nx_{m}.zip"); n = [x for x in z.namelist() if "NIFTY_500_" in x][0]
    syms, buf = [], ""
    for p in pypdf.PdfReader(io.BytesIO(z.read(n))).pages:
        for line in p.extract_text().splitlines():
            buf = line if not buf else buf + " " + line
            if num.search(buf):
                mm = sym.match(buf)
                if mm: syms.append(mm.group(1))
                buf = ""
            elif not sym.match(buf): buf = ""
    lists[m] = list(dict.fromkeys(syms)); print(m, len(lists[m]))
keys = list(snaps); out = []
for t in dict.fromkeys(s for m in keys for s in lists[m]):
    start = None
    for i, m in enumerate(keys):
        if t in lists[m] and start is None: start = snaps[m]
        if t not in lists[m] and start is not None: out.append((t, start, snaps[m])); start = None
    if start is not None: out.append((t, start, ""))
pd.DataFrame(out, columns=["ticker", "start", "end"]).to_csv(sys.argv[1], index=False)
print("tickers ever:", len({o[0] for o in out}), "intervals:", len(out))
