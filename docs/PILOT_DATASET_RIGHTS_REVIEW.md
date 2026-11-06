# Pilot dataset rights review

Prepared 2026-11-06 against `main` 058bf34. **This is an engineering and documentation review, not a
legal opinion, and nothing in it is an approval.** Formal authorization can only come from the
authorized reviewer named by rappidAI Research (docs/RIGHTS_REVIEW.md). The active
`data/garden/approvals.yaml` is empty and every registry source is still `production_approved: false`;
0 records are training-approved.

Machine-readable evidence: `data/garden/seed_v1/pilot/cohort.json`. Proposed approval templates (not
loadable, every decision field `<REQUIRED>`): `data/garden/seed_v1/pilot/proposed_approvals.yaml`.
Regenerate both with `python scripts/pilot_cohort.py --inspect`; `tests/test_pilot_cohort.py` checks
that they match the committed manifest, that the evidence covers the current source state, and that
they approve nothing.

## Executive summary

1. **Recommended repositories:** 48 of the 50 inspected pilot repositories (table below), 2 need
   clarification. Selected from the 510 Tier A repositories with records, not by size.
2. **Licenses:** MIT 31, Apache-2.0 14, BSD-2-Clause 1, BSD-3-Clause 1, ISC 1. All permissive, each
   license file identical at every revision the records use, and the package registry (deps.dev) agrees.
3. **Obligations:** none of the 48 has a NOTICE file or another license file at the root or on the paths
   of the used files. Their licenses require keeping the copyright and license notice **when the code
   is redistributed**. The plan is `not_redistributed` (code is rebuilt from upstream, never published);
   the advisory metadata (CC-BY-4.0) requires attribution to the GitHub Advisory Database.
4. **Uncertain:** whether a trained model's weights are a redistribution or adaptation of the training
   code or advisory text (unsettled law); GitHub's Acceptable Use Policies §7 on using information from
   the service; OSV's own service terms. Details in "Open legal questions".
5. **Smallest reasonable pilot:** the 48 repositories → **129 records** (train 103, validation 26) plus
   52 patch examples. Enough to validate the training pipeline end to end and to run a short
   calibration on a GPU; **not** enough to expect a measurably better security model. See "Volume".
6. **Decisions for the project owner:** listed in "Decisions required" (two source approvals with an
   attribution plan, one policy decision, then a yes/no per repository).
7. **After authorization:** five commands, all deterministic and offline except the rebuild (see
   "After authorization").

Legal conclusions are marked as such; everything else is an engineering finding from the evidence.

## Source-level dossier

All documents retrieved 2026-11-06; the sha256 values match the copies retrieved 2026-11-04 for the
three that docs/RIGHTS_REVIEW.md already recorded.

| Source ID | Actual provider | What we use | Rights evidence | Obligations |
| --- | --- | --- | --- | --- |
| `github-advisory-db` | GitHub, Inc. - GitHub Advisory Database (GHSA) | GHSA ID, aliases, CWE IDs, summary (≤ 300 chars), publication date, fix-commit reference. **Summaries become training targets** (finding evidence text); IDs and summaries are committed in `candidates.jsonl`. | [LICENSE.md](https://github.com/github/advisory-database/blob/main/LICENSE.md): CC-BY-4.0 (sha256 `9e5f1b3c…0cd94`). [GitHub Terms for Additional Products, "Advisory Database"](https://docs.github.com/en/site-policy/github-terms/github-terms-for-additional-products-and-features#advisory-database) (source sha256 `f285d929…ebf57d`): "licensed under the Creative Commons Attribution 4.0 license. The attribution term may be fulfilled by linking to … https://github.com/advisories or to individual … records" | Attribution (link to github.com/advisories in the model card and dataset documentation); indicate if the material was modified (summaries are used verbatim). CC-BY-4.0 has no share-alike or non-commercial term. |
| `osv` | Google - OSV.dev (aggregator; public export `gs://osv-vulnerabilities` and API) | Only `GHSA-*` records, read from the export bucket (discovery) and the API (audit). No other OSV source's content is used. | [OSV data sources](https://github.com/google/osv.dev/blob/master/docs/data.md) (sha256 `edcfab57…e96a22`) lists the GitHub Advisory Database as CC-BY 4.0; OSV code is Apache-2.0 ([LICENSE](https://github.com/google/osv.dev/blob/master/LICENSE), sha256 `cfc7749b…23d30`) - that covers software, not data | As `github-advisory-db` (the content is GHSA's). **Not established:** separate terms of service for the public bucket/API; none are published in the OSV documentation. |
| `ghsa-osv-fix-commits` | Each upstream project on GitHub (one per repository) | Function-level excerpts before and after the upstream fix commit and the function-level patch, fetched anonymously over git; **not committed, not uploaded, not redistributed** | Per-repository open-source license (table below). [GitHub ToS D.5/D.6](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service) (sha256 `7e2a7a73…3d09`): public repositories may be viewed and forked; further rights come from the repository's license ("inbound=outbound") | The repository license's conditions (notice retention on redistribution; Apache-2.0 §4 NOTICE and change statements). See per-repository table. |

**Advisory metadata is not covered by the code licenses.** A repository's MIT license says nothing
about the GHSA summary that labels its code; those summaries are CC-BY-4.0 by GitHub. Both approvals
are needed, and the governance gate requires both.

### Open legal questions (for the owner or a lawyer)

These are not resolved by this review; engineering cannot answer them.

1. **Model weights as redistribution.** Whether publishing weights trained on MIT/Apache code or on
   CC-BY-4.0 text triggers the notice/attribution obligations of those licenses is unsettled. The
   conservative practice is to attribute anyway: list the upstream repositories and the GitHub
   Advisory Database in the model card (`EXTERNAL_RESOURCES.json` already lists every repository and
   commit).
2. **GitHub Acceptable Use Policies §7** ([source](https://docs.github.com/en/site-policy/acceptable-use-policies/github-acceptable-use-policies#7-information-usage-restrictions),
   sha256 `8f22cce7…db44cf8d`): "Researchers may use public, non-personal information from the Service
   for research purposes, only if any publications resulting from that research are open access."
   Whether this applies to repository code fetched over git and used under the code's own license, or
   only to scraped/user information, is not clear from the text. An open-weight release with an
   open-access model card would be consistent with the research clause; a closed commercial use might
   not be. **Seek advice before any non-open release.**
3. **OSV service terms.** No terms beyond the per-source data licenses were found; if Google publishes
   any for the bucket or API, they need a check.
4. **Model output reproducing training code.** The model might reproduce short excerpts of training
   code in its answers. For permissive licenses this is low risk, but output-side attribution is not
   implemented.

## Selection method (engineering)

Starting set: the 510 Tier A repositories that have accepted train/validation records (permissive
license, identical license file at every used revision, deps.dev agrees). Filters, all from committed
audit data:

- at least one complete vulnerable/fixed pair; no group with weak advisory↔code correspondence;
- no repository that also has `temporal_quarantine` records (keeps the temporal evaluation clean);
- at most 20 records per repository (no single project dominates);
- quality score ≥ 3 per vulnerable group on average: strong (+2) / moderate (+1) correspondence, a fix
  pattern consistent with the CWE family (+1), localization ≤ 6 lines (+1), patch-eligible pair (+1),
  minus coarse CWEs and items the AI-assisted sample marked questionable;
- then per language and split, a round-robin over CWE families: Python, JavaScript and Go 10 train
  repositories each, TypeScript 8, plus 3 validation repositories per language. The frozen split is
  never changed: validation repositories were chosen among those already in `validation`.

Then each selected repository was inspected at **every** vulnerable and fixed revision its records
use, from the Git cache: root `NOTICE`/`COPYING`/`LICENSE*`/`PATENTS`/`THIRD_PARTY*` files, license
files in every directory on the path of a used file, and license fields in `package.json`,
`pyproject.toml`, `setup.cfg` and `setup.py`.

**Evidence integrity.** Each repository's evidence carries a `source_state_sha256` over its record IDs
with their vulnerable and fixed revisions and file paths, plus the inspected revision and path lists.
A rebuild without `--inspect` reuses cached evidence only when all of these match the current
manifest and every revision entry is well formed; otherwise the evidence is dropped and the repository
becomes `needs_clarification` until it is inspected again. Every failed Git read (a commit missing
from a partial cache, an unreadable tree, a missing license or metadata blob) is recorded in
`problems` with its revision and path and marks that revision `inspection: incomplete`; a failed read
is never taken as the absence of a file, and an incomplete inspection never yields
`recommended_for_review`. Final inspection (2026-11-06): all 50 repositories at all of their
revisions `complete`, 0 failed reads. Tests: `tests/test_pilot_inspection.py`, `tests/test_pilot_cohort.py`.

Confirmed for the 48 recommended repositories: no NOTICE file and no license file other than the root
license (or an identical copy of it) at any used revision; package metadata, where present, names the
same license. Findings that were checked and are **not** concerns: `taipy/gui/LICENSE` (the same
Apache-2.0 text), `packages/react-dev-utils/LICENSE` in create-react-app (MIT, same as root),
pannellum's `COPYING` (the MIT text again), reva's `license.hcl` (a dependency-license allowlist),
`license_view.go` in kubeclarity and `src/twisted/copyright.py` (source files).

Assumptions (not verified): that the license file in the repository root governs the excerpted
function where no other license file is on its path; that commit authors had the right to contribute
under the repository license; that deps.dev reflects the published package.

### Needs clarification (not recommended)

| Repository | Finding | What would resolve it |
| --- | --- | --- |
| aimhubio/aim | Root `LICENSE` is Apache-2.0, but `setup.py` at both used revisions declares `MIT License` | Check the project's intended license (README, PyPI page at the fixed version); if Apache-2.0 is authoritative, it can be reviewed like the others |
| patrickjuchli/basic-ftp | MIT, but `NOTICE.txt` states "This product includes software developed at The Apache Software Foundation" (Apache Commons Net) | Determine whether the excerpted functions derive from Apache Commons Net; if so, Apache-2.0 obligations also apply |

## Pilot cohort

| # | Repository | Lang / split | License (file sha256) | Records (vuln / hard neg. / abstain) | CWEs | Proposed decision |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | [free5gc/smf](https://github.com/free5gc/smf) | go / train | [Apache-2.0](https://github.com/free5gc/smf/blob/e23ce97565f285eb99eed153743c62bf4c767c6e/LICENSE) (`ed127da2459a`) | 1 / 1 / 1 | CWE-306 | `recommended_for_review` |
| 2 | [golang/glog](https://github.com/golang/glog) | go / train | [Apache-2.0](https://github.com/golang/glog/blob/b8741656e406e66d6992bc2c9575e460ecaa0ec2/LICENSE) (`73ba74dfaa52`) | 1 / 1 / 1 | CWE-377 | `recommended_for_review` |
| 3 | [huandu/facebook](https://github.com/huandu/facebook) | go / train | [MIT](https://github.com/huandu/facebook/blob/8b34431b91b32903c8821b1d7621bf81a029d8e4/LICENSE) (`cebbb247fbbd`) | 1 / 1 / 1 | CWE-209 | `recommended_for_review` |
| 4 | [kyverno/kyverno](https://github.com/kyverno/kyverno) | go / train | [Apache-2.0](https://github.com/kyverno/kyverno/blob/cbd7d4ca24de1c55396fc3295e9fc3215832be7c/LICENSE) (`c71d239df917`) | 2 / 2 / 2 | CWE-20, CWE-285 | `recommended_for_review` |
| 5 | [oauth2-proxy/oauth2_proxy](https://github.com/oauth2-proxy/oauth2_proxy) | go / train | [MIT](https://github.com/oauth2-proxy/oauth2_proxy/blob/a316f8a06f3c0ca2b5fc5fa18a91781b313607b2/LICENSE) (`89807acf2309`) | 1 / 1 / 1 | CWE-601 | `recommended_for_review` |
| 6 | [openclarity/kubeclarity](https://github.com/openclarity/kubeclarity) | go / train | [Apache-2.0](https://github.com/openclarity/kubeclarity/blob/1d1178840703a72d9082b7fc4aea0a3326c5d294/LICENSE) (`095a02629dc9`) | 1 / 1 / 0 | CWE-89 | `recommended_for_review` |
| 7 | [opencloud-eu/reva](https://github.com/opencloud-eu/reva) | go / train | [Apache-2.0](https://github.com/opencloud-eu/reva/blob/95aa2bc5d980eaf6cc134d75782b4f5ac7b36ae1/LICENSE) (`5df2a0d87d6c`) | 1 / 1 / 1 | CWE-22 | `recommended_for_review` |
| 8 | [schollz/rwtxt](https://github.com/schollz/rwtxt) | go / train | [MIT](https://github.com/schollz/rwtxt/blob/c09fb17375c4c47b49524c688288af1fe20e730a/LICENSE) (`212186b36a39`) | 1 / 1 / 1 | CWE-79 | `recommended_for_review` |
| 9 | [smallstep/certificates](https://github.com/smallstep/certificates) | go / train | [Apache-2.0](https://github.com/smallstep/certificates/blob/ffd31ac0a87e03b0224cb8363094bfe602242888/LICENSE) (`6fac0a40e050`) | 1 / 1 / 1 | CWE-129 | `recommended_for_review` |
| 10 | [sqls-server/sqls](https://github.com/sqls-server/sqls) | go / train | [MIT](https://github.com/sqls-server/sqls/blob/468a23fc89af89f632cc023a10c031e4bc781797/LICENSE) (`00aa9f139248`) | 1 / 1 / 1 | CWE-77 | `recommended_for_review` |
| 11 | [filebrowser/filebrowser](https://github.com/filebrowser/filebrowser) | go / validation | [Apache-2.0](https://github.com/filebrowser/filebrowser/blob/7ed1425115be602c2b23236c410098ea2d74b42f/LICENSE) (`4a1946819f24`) | 2 / 2 / 0 | CWE-203, CWE-732 | `recommended_for_review` |
| 12 | [goadesign/goa](https://github.com/goadesign/goa) | go / validation | [MIT](https://github.com/goadesign/goa/blob/70b5a199d0f813d74423993832c424e1fc73fb39/LICENSE) (`dc6856a49be0`) | 1 / 1 / 1 | CWE-22 | `recommended_for_review` |
| 13 | [icewhaletech/casaos-gateway](https://github.com/IceWhaleTech/CasaOS-Gateway) | go / validation | [Apache-2.0](https://github.com/IceWhaleTech/CasaOS-Gateway/blob/391dd7f0f239020c46bf057cfa25f82031fc15f7/LICENSE) (`c71d239df917`) | 1 / 1 / 0 | CWE-306 | `recommended_for_review` |
| 14 | [ahmedadelfahim/express-xss-sanitizer](https://github.com/AhmedAdelFahim/express-xss-sanitizer) | javascript / train | [MIT](https://github.com/AhmedAdelFahim/express-xss-sanitizer/blob/5623009ef11dcf095c163a38dea07b9cc22ad19f/LICENSE) (`2ec2ed6f68a6`) | 2 / 2 / 1 | CWE-1321, CWE-183 | `recommended_for_review` |
| 15 | [bmeck/node-cookiejar](https://github.com/bmeck/node-cookiejar) | javascript / train | [MIT](https://github.com/bmeck/node-cookiejar/blob/eaa00021caf6ae09449dde826108153b578348e5/LICENSE) (`3d51f096fd60`) | 1 / 1 / 0 | CWE-1333 | `recommended_for_review` |
| 16 | [expressjs/basic-auth-connect](https://github.com/expressjs/basic-auth-connect) | javascript / train | [MIT](https://github.com/expressjs/basic-auth-connect/blob/bac1e6a8530e1efd0028800b9b588a37adb0d203/LICENSE) (`04cca8117801`) | 1 / 1 / 0 | CWE-208 | `recommended_for_review` |
| 17 | [facebook/create-react-app](https://github.com/facebook/create-react-app) | javascript / train | [MIT](https://github.com/facebook/create-react-app/blob/f5e415f3a5b66f07dcc60aba1b445fa7cda97268/LICENSE) (`f657f99d3fb9`) | 1 / 1 / 1 | CWE-78 | `recommended_for_review` |
| 18 | [form-data/form-data](https://github.com/form-data/form-data) | javascript / train | [MIT](https://github.com/form-data/form-data/blob/3d1723080e6577a66f17f163ecd345a21d8d0fd0/License) (`e5b780d4f38d`) | 1 / 1 / 1 | CWE-330 | `recommended_for_review` |
| 19 | [hapijs/inert](https://github.com/hapijs/inert) | javascript / train | [BSD-2-Clause](https://github.com/hapijs/inert/blob/e8f99f94da4cb08e8032eda984761c3f111e3e82/LICENSE) (`416d44f354e8`) | 1 / 1 / 1 | CWE-22 | `recommended_for_review` |
| 20 | [herber/hekto](https://github.com/herber/hekto) | javascript / train | [MIT](https://github.com/herber/hekto/blob/1e5c75f8259ba0daf9b2600db3c246cda1934c46/LICENSE) (`b42f0daf5989`) | 1 / 1 / 0 | CWE-601 | `recommended_for_review` |
| 21 | [mpetroff/pannellum](https://github.com/mpetroff/pannellum) | javascript / train | [MIT](https://github.com/mpetroff/pannellum/blob/9391ef8da6a6a98c6a9f8c97f101adb900523681/COPYING) (`e20b384f83f8`) | 1 / 1 / 1 | CWE-79 | `recommended_for_review` |
| 22 | [tootallnate/plist.js](https://github.com/TooTallNate/plist.js) | javascript / train | [MIT](https://github.com/TooTallNate/plist.js/blob/96e2303d059e6be0c9e0c4773226d14b4758de52/LICENSE) (`3677dc86e6e5`) | 1 / 1 / 1 | CWE-1321 | `recommended_for_review` |
| 23 | [yonle/bostr](https://github.com/Yonle/bostr) | javascript / train | [BSD-3-Clause](https://github.com/Yonle/bostr/blob/49181f4ec9ae1472c6675cab56bbc01e723855af/LICENSE) (`7d3f85060fda`) | 1 / 1 / 0 | CWE-285 | `recommended_for_review` |
| 24 | [myshenin/aws-lambda-multipart-parser](https://github.com/myshenin/aws-lambda-multipart-parser) | javascript / validation | [MIT](https://github.com/myshenin/aws-lambda-multipart-parser/blob/56ccb03af4dddebc2b2defb348b6558783d5757e/LICENSE) (`921cbd9b449a`) | 1 / 1 / 0 | CWE-20 | `recommended_for_review` |
| 25 | [pillarjs/send](https://github.com/pillarjs/send) | javascript / validation | [MIT](https://github.com/pillarjs/send/blob/ae4f2989491b392ae2ef3b0015a019770ae65d35/LICENSE) (`25a328069fe7`) | 1 / 1 / 0 | CWE-79 | `recommended_for_review` |
| 26 | [vercel/ms](https://github.com/vercel/ms) | javascript / validation | [MIT](https://github.com/vercel/ms/blob/caae2988ba2a37765d055c4eee63d383320ee662/license.md) (`6652830c2607`) | 1 / 1 / 0 | CWE-1333 | `recommended_for_review` |
| 27 | [avaiga/taipy](https://github.com/Avaiga/taipy) | python / train | [Apache-2.0](https://github.com/Avaiga/taipy/blob/129fd407ffca49ee4ab853772c88d0c873e038dd/LICENSE) (`d018748ddcfc`) | 1 / 1 / 0 | CWE-22 | `recommended_for_review` |
| 28 | [cyanomiko/dcnnt-py](https://github.com/cyanomiko/dcnnt-py) | python / train | [MIT](https://github.com/cyanomiko/dcnnt-py/blob/b4021d784a97e25151a5353aa763a741e9a148f5/LICENSE) (`8b0fb4a9b7f4`) | 1 / 1 / 0 | CWE-77 | `recommended_for_review` |
| 29 | [ethyca/fides](https://github.com/ethyca/fides) | python / train | [Apache-2.0](https://github.com/ethyca/fides/blob/685bae61c203d29ed189f4b066a5223a9bb774c6/LICENSE) (`c71d239df917`) | 1 / 1 / 0 | CWE-338 | `recommended_for_review` |
| 30 | [illagrenan/django-make-app](https://github.com/illagrenan/django-make-app) | python / train | [MIT](https://github.com/illagrenan/django-make-app/blob/acd814433d1021aa8783362521b0bd151fdfc9d2/LICENSE) (`c2a3d7311b9e`) | 1 / 1 / 0 | CWE-94 | `recommended_for_review` |
| 31 | [inveniosoftware/invenio-drafts-resources](https://github.com/inveniosoftware/invenio-drafts-resources) | python / train | [MIT](https://github.com/inveniosoftware/invenio-drafts-resources/blob/039b0cff1ad4b952000f4d8c3a93f347108b6626/LICENSE) (`6b2388f171d5`) | 1 / 1 / 0 | CWE-862 | `recommended_for_review` |
| 32 | [pdfminer/pdfminer.six](https://github.com/pdfminer/pdfminer.six) | python / train | [MIT](https://github.com/pdfminer/pdfminer.six/blob/b808ee05dd7f0c8ea8ec34bdf394d40e63501086/LICENSE) (`f01e5d699a33`) | 1 / 1 / 1 | CWE-502 | `recommended_for_review` |
| 33 | [piccolo-orm/piccolo](https://github.com/piccolo-orm/piccolo) | python / train | [MIT](https://github.com/piccolo-orm/piccolo/blob/edcfe3568382922ba3e3b65896e6e7272f972261/LICENSE) (`cc52298bed7a`) | 1 / 1 / 0 | CWE-203 | `recommended_for_review` |
| 34 | [qiskit/qiskit-ibm-runtime](https://github.com/Qiskit/qiskit-ibm-runtime) | python / train | [Apache-2.0](https://github.com/Qiskit/qiskit-ibm-runtime/blob/b78fca114133051805d00043a404b25a33835f4d/LICENSE.txt) (`7b7de08c64b1`) | 2 / 2 / 0 | CWE-502, CWE-627 | `recommended_for_review` |
| 35 | [reflex-dev/reflex](https://github.com/reflex-dev/reflex) | python / train | [Apache-2.0](https://github.com/reflex-dev/reflex/blob/ade12549f3c0ddab3d7382c581bc04a3c1f989ec/LICENSE) (`770df32eba7d`) | 1 / 1 / 1 | CWE-601 | `recommended_for_review` |
| 36 | [twisted/twisted](https://github.com/twisted/twisted) | python / train | [MIT](https://github.com/twisted/twisted/blob/046a164f89a0f08d3239ecebd750360f8914df33/LICENSE) (`78d059391ddd`) | 1 / 1 / 1 | CWE-79 | `recommended_for_review` |
| 37 | [iterative/pydrive2](https://github.com/iterative/PyDrive2) | python / validation | [Apache-2.0](https://github.com/iterative/PyDrive2/blob/c57355dc2033ad90b7050d681b2c3ba548ff0004/LICENSE) (`57763138117d`) | 1 / 1 / 0 | CWE-502 | `recommended_for_review` |
| 38 | [prefecthq/fastmcp](https://github.com/PrefectHQ/fastmcp) | python / validation | [Apache-2.0](https://github.com/PrefectHQ/fastmcp/blob/40bdfb6b1de0ce30609ee9ba5bb95ecd04a9fb71/LICENSE) (`43070e2d4e53`) | 1 / 1 / 1 | CWE-918 | `recommended_for_review` |
| 39 | [badopcode/nodash](https://github.com/BadOPCode/NoDash) | typescript / train | [MIT](https://github.com/BadOPCode/NoDash/blob/b9cc2b3b49f6cd5228e406bc57e17a28b998fea5/LICENSE) (`e6bcd6324bc1`) | 1 / 1 / 1 | CWE-1321 | `recommended_for_review` |
| 40 | [microsoft/kiota-typescript](https://github.com/microsoft/kiota-typescript) | typescript / train | [MIT](https://github.com/microsoft/kiota-typescript/blob/09f8bd9b34d68bf412a9b78f6ca7e7961ef14974/LICENSE) (`c2cfccb812fe`) | 1 / 1 / 0 | CWE-178 | `recommended_for_review` |
| 41 | [nextauthjs/next-auth](https://github.com/nextauthjs/next-auth) | typescript / train | [ISC](https://github.com/nextauthjs/next-auth/blob/6e15bdcb2d93c1ad5ee3889f702607637e79db50/LICENSE) (`523848dd40e2`) | 1 / 1 / 0 | CWE-290 | `recommended_for_review` |
| 42 | [octokit/plugin-paginate-rest.js](https://github.com/octokit/plugin-paginate-rest.js) | typescript / train | [MIT](https://github.com/octokit/plugin-paginate-rest.js/blob/bb6c4f945d8023902cf387391d2b2209261044ab/LICENSE) (`8cd41c03a45e`) | 1 / 1 / 0 | CWE-1333 | `recommended_for_review` |
| 43 | [secreto31126/whatsapp-api-js](https://github.com/Secreto31126/whatsapp-api-js) | typescript / train | [MIT](https://github.com/Secreto31126/whatsapp-api-js/blob/56620c65126427496a94d176082fbd8393a95b6d/LICENSE) (`d5c7cf07f761`) | 1 / 1 / 1 | CWE-347 | `recommended_for_review` |
| 44 | [sveltejs/svelte](https://github.com/sveltejs/svelte) | typescript / train | [MIT](https://github.com/sveltejs/svelte/blob/a31dec5eb30978cff7ff4d77f4bf316841f711bc/LICENSE.md) (`97a941bae1c5`) | 1 / 1 / 0 | CWE-79 | `recommended_for_review` |
| 45 | [urigo/graphql-mesh](https://github.com/Urigo/graphql-mesh) | typescript / train | [MIT](https://github.com/Urigo/graphql-mesh/blob/95d93e7c140c2995b37e9d822aa3fe4e24ed2e78/LICENSE) (`c0c5f43383ea`) | 1 / 1 / 0 | CWE-22 | `recommended_for_review` |
| 46 | [zcaceres/markdownify-mcp](https://github.com/zcaceres/markdownify-mcp) | typescript / train | [MIT](https://github.com/zcaceres/markdownify-mcp/blob/0284aa8f34d32c65e20d8cda2d429b7943c9af03/LICENSE) (`fd8dd0549a48`) | 1 / 1 / 0 | CWE-918 | `recommended_for_review` |
| 47 | [asymmetric-effort/nogginlessdom](https://github.com/asymmetric-effort/NogginLessDom) | typescript / validation | [MIT](https://github.com/asymmetric-effort/NogginLessDom/blob/25a3cbac665fae5663f8b71c073b80c3152dbe7b/LICENSE.txt) (`060b54c83b60`) | 1 / 1 / 1 | CWE-1333 | `recommended_for_review` |
| 48 | [kapilduraphe/mcp-watch](https://github.com/kapilduraphe/mcp-watch) | typescript / validation | [MIT](https://github.com/kapilduraphe/mcp-watch/blob/e7da78c5b4b960f8b66c254059ad9ebc544a91a6/LICENSE) (`2ade7403edfc`) | 1 / 1 / 1 | CWE-78 | `recommended_for_review` |
| 49 | [aimhubio/aim](https://github.com/aimhubio/aim) | python / validation | [Apache-2.0](https://github.com/aimhubio/aim/blob/b9e53df5e32d14bbd3a2c738e2db7187fb531e93/LICENSE) (`c71d239df917`) | 1 / 1 / 0 | CWE-22 | `needs_clarification` |
| 50 | [patrickjuchli/basic-ftp](https://github.com/patrickjuchli/basic-ftp) | typescript / validation | [MIT](https://github.com/patrickjuchli/basic-ftp/blob/2ecc8e2c500c5234115f06fd1dbde1aa03d70f4b/LICENSE.txt) (`5b417c1f5ee9`) | 2 / 1 / 1 | CWE-22, CWE-93 | `needs_clarification` |

Per-repository details (revisions, record IDs, advisories, quality signals, confirmed evidence, open
questions, the raw file hashes per revision) are in `cohort.json`.

## Volume and composition (if the 48 recommended repositories and the three sources were approved)

Computed by running the real governance gate in memory with hypothetical approvals; nothing is
written and no export is created.

| | Train | Validation | Total |
| --- | ---: | ---: | ---: |
| Vulnerable (`finding`) | 41 | 11 | 52 |
| Hard negative (`no_finding`) | 41 | 11 | 52 |
| Abstention (`abstain`) | 21 | 4 | 25 |
| **Detection examples** | **103** | **26** | **129** |
| Patch examples (opt-in task) | 41 | 11 | 52 |

- Languages: Go 41, JavaScript 34, Python 30, TypeScript 24 records.
- 28 distinct CWEs in 13 families: authentication/authorization 7, deserialization/input validation 6,
  path traversal 5, SSRF/open redirect 5, ReDoS/resource limits 5, XSS 5, command injection 4,
  crypto/TLS/integrity 4, information exposure 4, prototype pollution 3, other 2, code injection 1,
  SQL injection 1.
- Quality: advisory↔code correspondence strong for 50 of 52 vulnerable groups (2 moderate);
  localization median 1 line, max 8; 2 findings carry a coarse CWE-20 (kyverno, aws-lambda-multipart-
  parser). 4 of the 64 independent human-review items fall into the cohort (none reviewed yet).
- Length: with the real Qwen3.5-9B tokenizer, detection examples have median 791 and max 2,315 tokens,
  patch examples median 796 and max 2,273: all fit `max_seq_len` 4,096, none would be dropped.
  Detection examples total about 118,000 tokens.
- Separation: every record is accepted, i.e. it passed deduplication, near-duplicate and label-conflict
  checks and the contamination check against QuickEval, the public seed and `coding-v1`; the split is
  the frozen repository-level split; no holdout or quarantine record is in the cohort, and no cohort
  repository has quarantine records.

**Assessment.** 103 training examples are enough to prove the pipeline (governance gate, integrity
checks, QLoRA on the real model, checkpointing, evaluation against the baseline) and to calibrate
memory and throughput. They are too few to expect a reliable gain in detection quality, and the
validation split (26) is too small for fine-grained conclusions. Gaps: SQL injection and code injection
have one example each; TypeScript is the smallest language. Justified expansion after the pilot: the
remaining Tier A repositories in the existing review batches (about 1,100 more train/validation
records), reviewed batch by batch with the same procedure. No examples should be synthesized to fill
gaps.

## Decisions required from the project owner

1. **Source `github-advisory-db`:** approve for training under CC-BY-4.0 with attribution (model card
   and dataset card link https://github.com/advisories; the per-record GHSA links already exist). Yes/no.
2. **Source `osv`:** approve as the access path for GHSA records (no separate content). Yes/no; decide
   whether its service terms need a further check.
3. **Source `ghsa-osv-fix-commits`:** approve the upstream-code source, with rights decided per
   repository and `redistribution_status: not_redistributed`. Yes/no.
4. **Policy on GitHub AUP §7 and weight publication** (open questions 1 and 2): proceed with an
   open-weight, open-access release and attribution, or seek legal advice first. This is the one
   question that may warrant a lawyer.
5. **Per repository (48):** confirm each license by opening the linked license file (one click per
   row; the evidence for NOTICE files and nested licenses is already collected) and set `approved_for_training`
   or `blocked`. Decide whether to clarify or drop aimhubio/aim and patrickjuchli/basic-ftp.

## After authorization

1. Sources: in `data/registry.yaml`, for the three sources set `production_approved: true`,
   `license_status: reviewed`, `terms_status: approved_for_project`, an immutable `revision` (the
   OSV export date or GHSA commit you reviewed) and `reviewer: <your name>`.
2. Repositories: copy the entries you decided from `data/garden/seed_v1/pilot/proposed_approvals.yaml`
   into `data/garden/approvals.yaml`, replacing every `<REQUIRED>` value (reviewer = your name, date).
3. Rebuild the records (only on a fresh checkout; needs network to github.com) and export:

```bash
python -m sentinel.garden reconstruct --config configs/garden/seed_v1.yaml
```

```bash
python -m sentinel.garden curate --config configs/garden/seed_v1.yaml
```

```bash
python -m sentinel.garden check --config configs/garden/seed_v1.yaml && python -m sentinel.foundation
```

```bash
python -m sentinel.train --dry-run
```

`curate` exports exactly the approved records (expected: 103 train / 26 validation, record-ID list
hash `3dfde06f…f93ab8` in `cohort.json`); `--dry-run` re-checks every approval, hash and split
without loading a model. Training itself still needs approved GPU time and is not part of this review.
