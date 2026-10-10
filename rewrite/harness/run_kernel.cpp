// run_kernel: load a code object, build a kernarg block from a JSON spec, launch one kernel, dump buffers.
//
// usage: run_kernel --co FILE.{hsaco,co,hip} --spec SPEC.json --out DIR [--kernel NAME]
//   .hip files are compiled with `hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151` into DIR first.
//
// spec (sizes are bytes):
//   kernel        mangled name (or the rewrite's extern "C" name); --kernel overrides
//   grid, block   [x,y,z] in blocks / threads (hipModuleLaunchKernel semantics)
//   lds_bytes     dynamic LDS bytes (default 0)
//   kernarg_size  total kernarg block bytes (zero padded)
//   hidden        optional {"block_count_x":off, "group_size_x":off, "remainder_x":off,
//                 "global_offset_x":off, "grid_dims":off}, filled from grid/block
//   buffers       {name: {"bytes":N, "fill":FILL}}   FILL = "zero" | {"kind":...}
//       kinds: zero | const(value) | iota(start,step) | rand(seed,lo,hi) | file(path)
//       all but zero/file take "dtype": f32 bf16 f16 f64 i8 u8 i16 u16 i32 u32 i64 u64
//   args          [{"offset":O, "buffer":NAME, "byte_offset":B} |
//                  {"offset":O, "type":T, "value":V} | {"offset":O, "hex":"aabb.."}]
//   dump          [buffer names] written to DIR/NAME.bin after the launch
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// ---- minimal JSON ----
struct J {
  enum T { Null, Bool, Num, Str, Arr, Obj } t = Null;
  double n = 0; bool b = false; std::string s;
  std::vector<J> a; std::map<std::string, J> o;
  bool has(const std::string& k) const { return t == Obj && o.count(k); }
  const J& at(const std::string& k) const {
    auto it = o.find(k);
    if (it == o.end()) { fprintf(stderr, "spec: missing key '%s'\n", k.c_str()); exit(2); }
    return it->second;
  }
  long long i() const { return (long long)n; }
};
struct P {
  const std::string& s; size_t p = 0;
  explicit P(const std::string& x) : s(x) {}
  void ws() { while (p < s.size() && isspace((unsigned char)s[p])) p++; }
  [[noreturn]] void die(const char* m) { fprintf(stderr, "json: %s at %zu\n", m, p); exit(2); }
  std::string str() {
    std::string r; p++;
    while (p < s.size() && s[p] != '"') {
      if (s[p] == '\\') { p++; char c = s[p]; r += c == 'n' ? '\n' : c == 't' ? '\t' : c; }
      else r += s[p];
      p++;
    }
    if (p >= s.size()) die("unterminated string");
    p++; return r;
  }
  J val() {
    ws(); J v;
    if (p >= s.size()) die("eof");
    char c = s[p];
    if (c == '{') {
      v.t = J::Obj; p++; ws();
      if (s[p] == '}') { p++; return v; }
      for (;;) {
        ws(); std::string k = str(); ws();
        if (s[p++] != ':') die("expected ':'");
        v.o[k] = val(); ws();
        if (s[p] == ',') { p++; continue; }
        if (s[p] == '}') { p++; break; }
        die("object");
      }
    } else if (c == '[') {
      v.t = J::Arr; p++; ws();
      if (s[p] == ']') { p++; return v; }
      for (;;) {
        v.a.push_back(val()); ws();
        if (s[p] == ',') { p++; continue; }
        if (s[p] == ']') { p++; break; }
        die("array");
      }
    } else if (c == '"') { v.t = J::Str; v.s = str(); }
    else if (!s.compare(p, 4, "true")) { v.t = J::Bool; v.b = true; p += 4; }
    else if (!s.compare(p, 5, "false")) { v.t = J::Bool; p += 5; }
    else if (!s.compare(p, 4, "null")) { p += 4; }
    else {
      char* e; v.t = J::Num; v.n = strtod(s.c_str() + p, &e);
      if (e == s.c_str() + p) die("value");
      p = e - s.c_str();
    }
    return v;
  }
};

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  fprintf(stderr, "HIP error %s at line %d: %s\n", hipGetErrorName(e_), __LINE__, #x); exit(3); } } while (0)

// ---- element encoding ----
static uint16_t f2bf(float f) {  // round to nearest even, NaN kept quiet
  uint32_t u; memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40);
  return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1)) >> 16);
}
static uint16_t f2h(float f) { _Float16 h = (_Float16)f; uint16_t u; memcpy(&u, &h, 2); return u; }

static int dsize(const std::string& d) {
  if (d == "f32" || d == "i32" || d == "u32") return 4;
  if (d == "bf16" || d == "f16" || d == "i16" || d == "u16") return 2;
  if (d == "i8" || d == "u8") return 1;
  if (d == "i64" || d == "u64" || d == "f64") return 8;
  fprintf(stderr, "bad dtype %s\n", d.c_str()); exit(2);
}
static void put(uint8_t* dst, const std::string& d, double v) {
  if (d == "f32") { float f = (float)v; memcpy(dst, &f, 4); }
  else if (d == "f64") memcpy(dst, &v, 8);
  else if (d == "bf16") { uint16_t u = f2bf((float)v); memcpy(dst, &u, 2); }
  else if (d == "f16") { uint16_t u = f2h((float)v); memcpy(dst, &u, 2); }
  else if (d == "i8" || d == "u8") { int8_t x = (int8_t)(long long)v; memcpy(dst, &x, 1); }
  else if (d == "i16" || d == "u16") { int16_t x = (int16_t)(long long)v; memcpy(dst, &x, 2); }
  else if (d == "i32" || d == "u32") { int32_t x = (int32_t)(long long)v; memcpy(dst, &x, 4); }
  else { int64_t x = (int64_t)v; memcpy(dst, &x, 8); }
}
// splitmix64: simple, reproducible from python
struct Rng {
  uint64_t s;
  uint64_t next() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
};

static std::vector<uint8_t> fill(const J& f, size_t bytes) {
  std::vector<uint8_t> v(bytes, 0);
  std::string kind = f.t == J::Str ? f.s : f.at("kind").s;
  if (kind == "zero") return v;
  if (kind == "file") {
    std::ifstream in(f.at("path").s, std::ios::binary);
    if (!in) { fprintf(stderr, "cannot open fill file\n"); exit(2); }
    in.read((char*)v.data(), bytes); return v;
  }
  std::string d = f.at("dtype").s; int w = dsize(d);
  size_t cnt = bytes / w;
  Rng r{(uint64_t)(f.has("seed") ? f.at("seed").i() : 0)};
  for (size_t k = 0; k < cnt; k++) {
    double x = 0;
    if (kind == "const") x = f.at("value").n;
    else if (kind == "iota") x = f.at("start").n + k * (f.has("step") ? f.at("step").n : 1.0);
    else if (kind == "rand") {
      double lo = f.at("lo").n, hi = f.at("hi").n; uint64_t u = r.next();
      bool isint = d[0] == 'i' || d[0] == 'u';
      if (isint) x = lo + (double)(u % (uint64_t)(hi - lo));
      else x = lo + ((double)(u >> 11) * (1.0 / 9007199254740992.0)) * (hi - lo);
    } else { fprintf(stderr, "bad fill kind %s\n", kind.c_str()); exit(2); }
    put(v.data() + k * w, d, x);
  }
  return v;
}

static std::string slurp(const std::string& p) {
  std::ifstream in(p);
  if (!in) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(2); }
  std::stringstream ss; ss << in.rdbuf(); return ss.str();
}
static bool ends(const std::string& s, const std::string& e) {
  return s.size() >= e.size() && !s.compare(s.size() - e.size(), e.size(), e);
}

int main(int argc, char** argv) {
  std::string co, specp, out, kname;
  for (int k = 1; k + 1 < argc; k += 2) {
    std::string a = argv[k];
    if (a == "--co") co = argv[k + 1];
    else if (a == "--spec") specp = argv[k + 1];
    else if (a == "--out") out = argv[k + 1];
    else if (a == "--kernel") kname = argv[k + 1];
    else { fprintf(stderr, "unknown flag %s\n", a.c_str()); return 2; }
  }
  if (co.empty() || specp.empty() || out.empty()) {
    fprintf(stderr, "usage: run_kernel --co F --spec S --out DIR [--kernel N]\n"); return 2;
  }
  std::string stext = slurp(specp); P pp(stext); J spec = pp.val();
  if (kname.empty()) kname = spec.at("kernel").s;
  if (system(("mkdir -p '" + out + "'").c_str()) != 0) return 2;
  if (ends(co, ".hip")) {
    std::string obj = out + "/rewrite.co";
    std::string cmd = "hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o '" + obj + "' '" + co + "' 2>&1";
    if (system(cmd.c_str()) != 0) { fprintf(stderr, "hipcc failed\n"); return 4; }
    co = obj;
  }
  struct Buf { void* d; size_t bytes; };
  std::map<std::string, Buf> bufs; size_t total = 0;
  for (auto& kv : spec.at("buffers").o) total += (size_t)kv.second.at("bytes").i();
  if (total > (2ull << 30)) { fprintf(stderr, "buffers %zu bytes exceed the 2 GiB cap\n", total); return 5; }
  CK(hipSetDevice(0));
  hipModule_t mod; CK(hipModuleLoad(&mod, co.c_str()));
  hipFunction_t fn; CK(hipModuleGetFunction(&fn, mod, kname.c_str()));
  for (auto& kv : spec.at("buffers").o) {
    size_t bytes = (size_t)kv.second.at("bytes").i();
    auto host = fill(kv.second.at("fill"), bytes);
    void* d; CK(hipMalloc(&d, bytes ? bytes : 1));
    if (bytes) CK(hipMemcpy(d, host.data(), bytes, hipMemcpyHostToDevice));
    bufs[kv.first] = {d, bytes};
  }
  size_t ksz = (size_t)spec.at("kernarg_size").i();
  std::vector<uint8_t> ka(ksz, 0);
  auto room = [&](size_t off, size_t n) {
    if (off + n > ksz) { fprintf(stderr, "arg at %zu+%zu beyond kernarg_size %zu\n", off, n, ksz); exit(2); }
  };
  for (auto& a : spec.at("args").a) {
    size_t off = (size_t)a.at("offset").i();
    if (a.has("buffer")) {
      auto it = bufs.find(a.at("buffer").s);
      if (it == bufs.end()) { fprintf(stderr, "unknown buffer %s\n", a.at("buffer").s.c_str()); return 2; }
      uint64_t ptr = (uint64_t)(uintptr_t)it->second.d + (a.has("byte_offset") ? a.at("byte_offset").i() : 0);
      room(off, 8); memcpy(&ka[off], &ptr, 8);
    } else if (a.has("hex")) {
      const std::string& h = a.at("hex").s;
      room(off, h.size() / 2);
      for (size_t k = 0; k + 1 < h.size(); k += 2) ka[off + k / 2] = (uint8_t)strtoul(h.substr(k, 2).c_str(), nullptr, 16);
    } else {
      std::string t = a.at("type").s; room(off, dsize(t));
      put(&ka[off], t, a.at("value").n);
    }
  }
  auto g = spec.at("grid").a, bl = spec.at("block").a;
  if (g.size() != 3 || bl.size() != 3) { fprintf(stderr, "grid and block need 3 values each\n"); return 2; }
  if (spec.has("hidden")) {
    const J& h = spec.at("hidden");
    auto w32 = [&](const std::string& k, uint32_t v) { if (h.has(k)) { room(h.at(k).i(), 4); memcpy(&ka[h.at(k).i()], &v, 4); } };
    auto w16 = [&](const std::string& k, uint16_t v) { if (h.has(k)) { room(h.at(k).i(), 2); memcpy(&ka[h.at(k).i()], &v, 2); } };
    auto w64 = [&](const std::string& k, uint64_t v) { if (h.has(k)) { room(h.at(k).i(), 8); memcpy(&ka[h.at(k).i()], &v, 8); } };
    const char* ax = "xyz";
    for (int k = 0; k < 3; k++) {
      std::string s(1, ax[k]);
      w32("block_count_" + s, (uint32_t)g[k].i());
      w16("group_size_" + s, (uint16_t)bl[k].i());
      w16("remainder_" + s, 0);
      w64("global_offset_" + s, 0);
    }
    w16("grid_dims", (uint16_t)((g[2].i() > 1 || bl[2].i() > 1) ? 3 : (g[1].i() > 1 || bl[1].i() > 1) ? 2 : 1));
  }
  size_t kasz = ksz;
  void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, ka.data(), HIP_LAUNCH_PARAM_BUFFER_SIZE, &kasz, HIP_LAUNCH_PARAM_END};
  unsigned lds = spec.has("lds_bytes") ? (unsigned)spec.at("lds_bytes").i() : 0;
  // Pre-launch copy of every dumped buffer: compare.py fails a case whose output the kernel never wrote.
  if (spec.has("dump")) for (auto& nm : spec.at("dump").a) {
    auto it = bufs.find(nm.s);
    if (it == bufs.end()) { fprintf(stderr, "dump: unknown buffer %s\n", nm.s.c_str()); return 2; }
    std::vector<uint8_t> h(it->second.bytes);
    if (!h.empty()) CK(hipMemcpy(h.data(), it->second.d, h.size(), hipMemcpyDeviceToHost));
    std::ofstream o(out + "/" + nm.s + ".pre.bin", std::ios::binary);
    o.write((char*)h.data(), h.size());
  }
  CK(hipModuleLaunchKernel(fn, g[0].i(), g[1].i(), g[2].i(), bl[0].i(), bl[1].i(), bl[2].i(), lds, nullptr, nullptr, cfg));
  CK(hipDeviceSynchronize());
  CK(hipGetLastError());
  if (spec.has("dump")) for (auto& nm : spec.at("dump").a) {
    auto it = bufs.find(nm.s);
    if (it == bufs.end()) { fprintf(stderr, "dump: unknown buffer %s\n", nm.s.c_str()); return 2; }
    std::vector<uint8_t> h(it->second.bytes);
    if (!h.empty()) CK(hipMemcpy(h.data(), it->second.d, h.size(), hipMemcpyDeviceToHost));
    std::ofstream o(out + "/" + nm.s + ".bin", std::ios::binary);
    o.write((char*)h.data(), h.size());
  }
  for (auto& kv : bufs) (void)hipFree(kv.second.d);
  CK(hipModuleUnload(mod));
  printf("ok kernel=%s buffers=%zu bytes\n", kname.c_str(), total);
  return 0;
}
