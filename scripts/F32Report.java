// Post-analysis report: disassembly coverage + decompiler smoke test for F32 module.
// Ghidra auto-compiles this with the bundled Eclipse JDT at script runtime.
// @category Analysis
import ghidra.app.script.GhidraScript;
import ghidra.app.decompiler.*;
import ghidra.program.model.listing.*;
import ghidra.program.model.address.*;

public class F32Report extends GhidraScript {
    @Override
    public void run() throws Exception {
        Listing listing = currentProgram.getListing();
        Address start = currentProgram.getMinAddress();
        Address end = currentProgram.getMaxAddress();
        InstructionIterator it = listing.getInstructions(true);
        int ins = 0, data = 0, undef = 0;
        StringBuilder sample = new StringBuilder();
        while (it.hasNext() && ins < 100000) {
            Instruction in = it.next();
            ins++;
            if (ins <= 12) sample.append(in.toString()).append('\n');
        }
        DataIterator di = listing.getData(true);
        while (di.hasNext()) {
            Data d = di.next();
            if (d.isDefined()) data++; else undef++;
        }
        int fns = currentProgram.getFunctionManager().getFunctionCount();
        StringBuilder out = new StringBuilder();
        out.append("F32REPORT language=").append(currentProgram.getLanguage().getLanguageID().toString())
           .append(" ins=").append(ins).append(" definedData=").append(data)
           .append(" undef=").append(undef).append(" functions=").append(fns).append('\n');
        out.append(sample);
        // decompiler smoke test on first function
        FunctionIterator fits = currentProgram.getFunctionManager().getFunctions(true);
        if (fits.hasNext()) {
            Function f = fits.next();
            DecompInterface decomp = new DecompInterface();
            decomp.openProgram(currentProgram);
            DecompileResults r = decomp.decompileFunction(f, 60, monitor);
            out.append("DECOMP function=").append(f.getName()).append('@').append(f.getEntryPoint())
               .append(" ok=").append(r.decompileCompleted()).append('\n');
            if (r.decompileCompleted()) {
                String c = r.getDecompiledFunction().getC();
                out.append(c.length() > 1200 ? c.substring(0, 1200) : c).append('\n');
            } else {
                out.append("DECOMPERR ").append(r.getErrorMessage()).append('\n');
            }
        }
        java.io.File f = new java.io.File(System.getProperty("f32.report", "f32report.txt"));
        java.nio.file.Files.write(f.toPath(), out.toString().getBytes());
        println(out.toString());
    }
}
