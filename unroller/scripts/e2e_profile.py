"""Profile one pipeline stage in a fresh process: wall, CPU utilisation, peak RSS, bytes fetched, cProfile top-10.
usage: python e2e.py OUT.json NAME -- cli args...   |   python e2e.py OUT.json fibers ZARR"""
import cProfile, io, json, os, pstats, resource, sys, time
out_json, name = sys.argv[1], sys.argv[2]
args = sys.argv[4:] if sys.argv[3] == "--" else sys.argv[3:]
import unroller.volume as uv
stores = []
_init = uv._Store.__init__
def init(self, *a, **k):
    _init(self, *a, **k); stores.append(self)
uv._Store.__init__ = init
def run():
    if name == "fibers":
        from unroller import store
        from unroller.fibers import fiber_maps
        st = store.load(args[0])
        n, H, W = st.shape
        r0, c0 = max((H - 1024) // 2, 0), max((W - 1024) // 2, 0)
        crop = st[:, r0:r0 + 1024, c0:c0 + 1024].astype("float32")
        print("fiber crop", crop.shape, file=sys.stderr)
        res = fiber_maps(crop)
        print("keys", list(res)[:8], file=sys.stderr)
    else:
        from unroller.cli import main
        main(args)
pr = cProfile.Profile()
r0 = resource.getrusage(resource.RUSAGE_SELF)
t = time.perf_counter()
pr.enable()
run()
pr.disable()
wall = time.perf_counter() - t
r1 = resource.getrusage(resource.RUSAGE_SELF)
cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(10)
top = [l for l in s.getvalue().splitlines() if l.strip()][-11:]
rec = {"stage": name, "wall_s": round(wall, 2), "cpu_s": round(cpu, 2), "cpu_util_of_4_cores": round(cpu / wall / 4, 2),
       "peak_rss_MB": round(r1.ru_maxrss / 1024, 1), "bytes_fetched_MB": round(sum(x.bytes_fetched for x in stores) / 1e6, 1),
       "http_requests": sum(x.requests for x in stores)}
print(json.dumps(rec))
with open(out_json, "a") as f:
    f.write(json.dumps({**rec, "top10_tottime": top}) + "\n")
pstats.Stats(pr).dump_stats(out_json + f".{name}.prof")
