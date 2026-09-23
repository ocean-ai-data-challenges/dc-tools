#!/usr/bin/env python3
"""Thread diagnostic for Dask workers."""
import os, time, subprocess, multiprocessing

for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
          'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','GOTO_NUM_THREADS',
          'BLOSC_NTHREADS','PYINTERP_NUM_THREADS'):
    os.environ[v] = '1'

multiprocessing.set_start_method('forkserver', force=True)

def count_threads():
    import os, ctypes
    pid = os.getpid()
    thr = len(os.listdir('/proc/%d/task' % pid))
    oblas = {}
    seen = set()
    with open('/proc/%d/maps' % pid) as f:
        for line in f:
            if 'openblas' not in line.lower(): continue
            parts = line.split()
            if len(parts) < 6: continue
            path = parts[-1]
            if not path.startswith('/') or path in seen: continue
            seen.add(path)
            try:
                lib = ctypes.CDLL(path)
                for sym in ('openblas_get_num_threads64_','openblas_get_num_threads','scipy_openblas_get_num_threads'):
                    try:
                        g = getattr(lib, sym); g.restype = ctypes.c_int
                        oblas[os.path.basename(path)[:30] + '/' + sym.split('_')[-1]] = g()
                    except AttributeError: pass
            except: pass
    return {'pid': pid, 'os_threads': thr, 'openblas': oblas}

def main():
    from dask.distributed import Client, LocalCluster

    print("=== Creating cluster: 5 workers, 1 thread each ===")
    cluster = LocalCluster(n_workers=5, threads_per_worker=1, processes=True, memory_limit='6GB')
    client = Client(cluster)
    time.sleep(3)

    # BEFORE cleanup
    print("\n--- BEFORE _worker_full_cleanup ---")
    info = client.run(count_threads)
    for addr, d in info.items():
        ob = " | ".join("%s=%d" % (k,v) for k,v in d['openblas'].items())
        print("  Worker pid=%d  os_threads=%d  %s" % (d['pid'], d['os_threads'], ob))

    # Run cleanup
    from dctools.metrics.evaluator import _worker_full_cleanup
    client.run(_worker_full_cleanup)
    time.sleep(1)

    # AFTER cleanup
    print("\n--- AFTER _worker_full_cleanup ---")
    info = client.run(count_threads)
    for addr, d in info.items():
        ob = " | ".join("%s=%d" % (k,v) for k,v in d['openblas'].items())
        print("  Worker pid=%d  os_threads=%d  %s" % (d['pid'], d['os_threads'], ob))

    # Total threads
    r = subprocess.run("ps -e -o nlwp,comm | grep python | awk '{s+=$1}END{print s}'",
                       shell=True, capture_output=True, text=True)
    print("\nTOTAL threads across ALL python processes: %s" % r.stdout.strip())

    # Main process
    main_thr = len(os.listdir('/proc/%d/task' % os.getpid()))
    print("Main process: os_threads=%d" % main_thr)

    # All python process details
    r2 = subprocess.run("ps -e -o pid,nlwp,rss,comm | grep python | grep -v grep",
                        shell=True, capture_output=True, text=True)
    print("\nAll python processes:")
    for line in r2.stdout.strip().split('\n'):
        print("  " + line.strip())

    # Now simulate work: run pyinterp in a worker
    print("\n=== Testing pyinterp (num_threads=1) in one worker ===")
    def pyinterp_test():
        import numpy as np, pyinterp, pyinterp.backends.xarray, xarray as xr, time as _t, os
        n_t = int(os.environ.get('PYINTERP_NUM_THREADS', '1'))
        lon = np.linspace(0,360,361); lat = np.linspace(-90,90,181)
        da = xr.DataArray(np.random.randn(len(lat),len(lon)).astype(np.float64),
                          dims=('lat','lon'), coords={'lat':lat,'lon':lon})
        grid = pyinterp.backends.xarray.Grid2D(da)
        obs_lon = np.random.uniform(0,360,500000); obs_lat = np.random.uniform(-90,90,500000)
        t0=_t.time(); c0=_t.process_time()
        for _ in range(5):
            pyinterp.bivariate(grid, x=obs_lon, y=obs_lat, interpolator='bilinear',
                              bounds_error=False, num_threads=max(1,n_t))
        w=_t.time()-t0; c=_t.process_time()-c0
        thr=len(os.listdir('/proc/%d/task'%os.getpid()))
        return "Wall:%.2f CPU:%.2f Util:%.0f%% os_thr:%d pyinterp_n:%d" % (w,c,c/w*100,thr,n_t)

    res = client.run(pyinterp_test)
    for addr, r in res.items():
        print("  %s: %s" % (addr, r))

    client.close(); cluster.close()
    print("\nDone.")

if __name__ == '__main__':
    main()
