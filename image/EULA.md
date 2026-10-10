# halogen — End User License Agreement

**Version 1.0 · Effective 10 October 2026**

This Agreement is between you ("**You**") and **Peonist, LLC** ("we", "us"),
and governs your use of **halogen** (the "**Software**") — the container
image published by us containing the halogen inference engine binary, its
serving front-end, and supporting scripts.

By downloading, running, or otherwise using the Software, you agree to this
Agreement. If you do not agree, do not use the Software.

---

## 1. What this covers, and what it does not

This Agreement covers **only** the parts of the Software that we wrote: the
halogen engine binary, the serving front-end, the benchmark tooling, and the
container packaging.

It does **not** cover:

- **Model weights.** The Software contains no model weights. Weights are
  obtained separately and are licensed separately, including by their original
  authors. Nothing here grants you any right to any model weights. It is your
  responsibility to comply with the terms attached to whatever weights you
  load.
- **Third-party components** redistributed inside the image — the AMD ROCm
  runtime and libraries, the Python interpreter, and Python packages. Each is
  governed by its own license, reproduced in `THIRD-PARTY-NOTICES.md` in this
  directory and inside the image. Where a third-party license conflicts with
  this Agreement, **the third-party license governs that component.**

## 2. Open-source license grant

The Software is **free and open source**. Subject to the conditions in this
Agreement, we grant you a worldwide, royalty-free, non-exclusive, perpetual,
irrevocable license to:

1. **Use and run** the Software for any purpose, **including commercial and
   production use**, on any number of machines, with no seat, core, request,
   or revenue limit, and no license key.
2. **Study, review, and modify** the Software, in source and object form, and
   to create derivative works.
3. **Redistribute** the Software, in source or object form, modified or
   unmodified, including by mirroring it into your own container registry.
4. **Extract, adapt, and reuse components** of the Software — source files,
   build tooling, scripts, packaging, and data formats — in your own
   projects, at your own free will and without additional permission.

No fee is payable and no registration is required.

**Conditions.** In exchange, copies of the Software and of any derivative
work must:

1. carry a prominent notice stating that they are derived from halogen and
   identifying its origin;
2. include the copyright notice, this Agreement, and
   `THIRD-PARTY-NOTICES.md` intact; and
3. clearly document any modifications, so that a modified copy is not
   represented as the unmodified Software.

## 3. Reverse engineering — expressly permitted, and encouraged

**Reverse engineering of the Software is explicitly permitted without
limitation.** You may decompile, disassemble, emulate, instrument, profile,
and otherwise analyze any part of the Software — including the engine binary,
its compiled gfx1151 device code and fatbin kernels, the serving front-end,
the `.hgn` container format, and any other component shipped in the image —
for any purpose whatsoever, including interoperability, security research,
debugging, academic study, teaching, and personal curiosity.

This permission is unconditional. It requires no separate authorization, no
notice, no registration, and no agreement beyond this one, and no provision
of this Agreement — including Section 4 — restricts it.

**You may publish what you find.** Writeups, specifications, disassemblies,
decompilations, recovered algorithms, re-implementations, ports, forks, and
derivative works of any kind are yours to distribute under terms of your
choosing. We claim no approval right over them and assert no copyright
against them, provided the attribution conditions in Section 2 are met.

**This is an open community challenge.** The Software is published as a
compiled binary by design, and we want it taken apart. Find out how the
kernels work. Recover the formats. Break the assumptions. Show us where the
engine is slow, wrong, unsafe, or leaving performance on the table, and
improve it. Fork it, rebuild it, and ship something better under your own
name. Anything you discover is a contribution to the project, not a
violation of it.

## 4. Restrictions

You may not:

1. **Use our names or marks** — "Peonist", "halogen" — to endorse or promote
   your products without our written permission. Factually stating that your
   product runs on halogen, is derived from halogen, or is compatible with
   halogen is fine and needs no permission. `TRADEMARKS.md` states the full
   policy.
2. **Present the Software as warrantied.** The disclaimers in Sections 6 and
   7 must travel with every copy and derivative.
3. **Misrepresent provenance.** Do not remove the attribution notices
   required by Section 2, and do not represent a modified copy as the
   unmodified Software.

## 5. Benchmarking and publication — expressly permitted

**You may run benchmarks on the Software and publish the results.** No
approval, notice, or prior review by us is required. We ship `bench` and
`sweep` modes in the image specifically so that you can.

We ask only that published figures state:

1. the Software version they were produced with, and
2. **which prompt set produced them.**

The second is a technical necessity, not a legal formality. halogen's decode
throughput depends on speculative-decoding acceptance, which depends on how
predictable the generated text is — the same build measured on our own prompt
set spans roughly 2x, from about 20 tokens/s on prose to about 42 on code.
A single number without a named prompt set is not reproducible and is not
meaningful, in either direction. This is a request, and not a condition of
your license: publishing a figure without naming the prompt set does not
breach this Agreement.

## 6. No telemetry

The Software performs no telemetry, no license checks, and no usage
reporting, and it makes no callback to us of any kind. **We receive no
information whatsoever about your use of it**, and there is no configuration
that changes that.

By default it opens **no outbound network connections at all** — it listens
for the requests you send it and talks to nothing else. There is exactly one
exception, and it is off unless you switch it on: if you set
`HALOGEN_DOWNLOAD`, the Software fetches model weights at startup from the
repository you name. That is a connection to a third-party host of your
choosing, not to us, and it happens only when the weights are absent.

## 7. No warranty

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT.

Stated concretely, because it matters for this kind of software: we do not
warrant that the Software will produce correct, accurate, or fit-for-purpose
output; that it will achieve any particular throughput on your hardware; that
it will run on any configuration other than AMD Strix Halo (gfx1151), for
which it is built exclusively; or that it will operate uninterrupted or
error-free. **Model output is generated text and may be wrong.** Do not rely
on it for safety-critical, medical, legal, or financial decisions without
independent verification.

## 8. Limitation of liability

TO THE MAXIMUM EXTENT PERMITTED BY LAW, IN NO EVENT WILL WE BE LIABLE FOR ANY
INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES, OR ANY LOSS
OF PROFITS, REVENUE, DATA, OR BUSINESS INTERRUPTION, ARISING OUT OF OR RELATED
TO THIS AGREEMENT OR THE SOFTWARE, WHETHER IN CONTRACT, TORT, OR OTHERWISE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

OUR TOTAL AGGREGATE LIABILITY UNDER THIS AGREEMENT WILL NOT EXCEED THE GREATER
OF (A) THE AMOUNT YOU PAID US FOR THE SOFTWARE, WHICH FOR THIS VERSION IS
ZERO, OR (B) FIFTY UNITS OF THE APPLICABLE CURRENCY.

Some jurisdictions do not allow the exclusion of certain warranties or
liabilities; in those places, the exclusions above apply only to the extent
permitted, and nothing here limits liability for fraud, or for death or
personal injury caused by negligence.

## 9. Term and termination

This license is perpetual and irrevocable. It terminates only if you breach
Section 4, and you must then stop distributing the Software in breach;
licenses you have already granted to third parties for copies they lawfully
received are unaffected. Sections 1, 7, 8, and 10 survive termination.

Your rights under Section 2 for a copy you already hold are not withdrawn by
us ceasing to publish the Software.

## 10. General

This Agreement is the entire agreement between us regarding the Software and
supersedes prior discussions. If any provision is held unenforceable, the rest
remains in effect. Our failure to enforce a provision is not a waiver of it.

This Agreement is governed by the laws of the **State of Florida, United
States**, without regard to its conflict-of-law rules.

We may publish future versions of the Software under different terms; those
terms will apply to those versions, not retroactively to this one.

---

**Contact:** open an issue at https://github.com/peonist-ai/halogen-flash-server
**Copyright © 2026 Peonist, LLC.**
