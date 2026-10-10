// Ghidra headless script: decompile every function to <outdir>/<addr>.c and write index.tsv.
// Usage: analyzeHeadless <proj> <name> -import <elf> -postScript ExportDecomp.java <outdir>
// index.tsv columns: addr, name, bytes, callees, callers, strings (up to 4, '|' joined)
import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.data.StringDataInstance;
import ghidra.program.model.listing.Data;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;
import ghidra.program.model.symbol.Reference;
import ghidra.util.task.ConsoleTaskMonitor;

import java.io.File;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.util.LinkedHashSet;
import java.util.Set;

public class ExportDecomp extends GhidraScript {
    @Override
    public void run() throws Exception {
        String[] args = getScriptArgs();
        File out = new File(args.length > 0 ? args[0] : "decomp");
        out.mkdirs();
        DecompInterface di = new DecompInterface();
        di.openProgram(currentProgram);
        int done = 0, failed = 0;
        try (PrintWriter idx = new PrintWriter(new File(out, "index.tsv"))) {
            idx.println("addr\tname\tbytes\tcallees\tcallers\tstrings");
            FunctionIterator it = currentProgram.getFunctionManager().getFunctions(true);
            while (it.hasNext() && !monitor.isCancelled()) {
                Function f = it.next();
                String addr = f.getEntryPoint().toString();
                Set<String> strs = new LinkedHashSet<>();
                for (Address a : f.getBody().getAddresses(true)) {
                    for (Reference r : currentProgram.getReferenceManager().getReferencesFrom(a)) {
                        Data d = currentProgram.getListing().getDataAt(r.getToAddress());
                        if (d != null && d.hasStringValue() && strs.size() < 4) {
                            String s = String.valueOf(d.getValue()).replace('\t', ' ').replace('\n', ' ');
                            strs.add(s.length() > 60 ? s.substring(0, 60) : s);
                        }
                    }
                    if (strs.size() >= 4) break;
                }
                DecompileResults res = di.decompileFunction(f, 90, new ConsoleTaskMonitor());
                if (res != null && res.decompileCompleted()) {
                    Files.writeString(new File(out, addr + ".c").toPath(), res.getDecompiledFunction().getC());
                    done++;
                } else {
                    failed++;
                }
                idx.println(addr + "\t" + f.getName() + "\t" + f.getBody().getNumAddresses() + "\t"
                    + f.getCalledFunctions(monitor).size() + "\t" + f.getCallingFunctions(monitor).size()
                    + "\t" + String.join("|", strs));
                if ((done + failed) % 100 == 0) println("decompiled " + (done + failed));
            }
        }
        println("EXPORT_DONE ok=" + done + " failed=" + failed);
    }
}
