# templates

Instruments templates for `../tools/xctrace_record.py`: when `kprof.tracetemplate` exists here, recordings use it instead of the stock Metal System Trace, so they include per-shader intervals and GPU hardware counters.

Instruments only turns on the shader timeline and GPU counter sets in a template saved from its
GUI; a template edited by hand is rejected ("Selected counter profile is not supported on target
device"). Counter sets are specific to a GPU family, so save the template on the machine (or
chip generation) it is used on:

1. Open Instruments (Xcode > Open Developer Tool > Instruments), pick **Metal System Trace**, and
   select this Mac as the target.
2. In the recording options of the GPU counters track, choose a counter set (performance
   limiters, memory, occupancy, ...: whatever this GPU offers) and turn on the shader timeline if
   it is offered.
3. File > Save as Template..., to `ios_kernels/profiling/templates/kprof.tracetemplate`.
4. Check it: record any `kbench` run with `xctrace_record.py`, then run `xctrace_export.py
   <trace> --counters`. The export lists shader intervals and counters other than
   "RT Unit Active".
