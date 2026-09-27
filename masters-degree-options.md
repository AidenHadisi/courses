# Affordable Online Master's Options for a CS Grad (Sept 2026)

Research into which master's degree to pursue as a working software engineer in California with a BS in Computer Science. The goal is a hedge against AI-driven disruption of software jobs, with total tuition under about $20k and a finish in roughly 12 months (slower options are listed too). School brand matters, but cost and speed win ties.

All prices are 2026–27 rates for a California resident, taken from each school's own pages unless noted. Several schools changed prices this year, so re-check before enrolling.

Companion doc: [certificates.md](./certificates.md) (industry certs, professional certificates, and stackable graduate certificates in cybersecurity and AI).

## TL;DR

- **The best hedge is to broaden into hardware or physical systems, not to double down on software or AI.** The evidence points to electrical and computer engineering (ECE): embedded systems, FPGAs, computer architecture, chip design (VLSI), and power systems. Hardware-adjacent security also qualifies.
- **The best-fit program is CU Boulder's online MS in Electrical & Computer Engineering (MS-ECE) on Coursera: $20,010.** There's no application. You're admitted by passing a 3–4 course "pathway," and the FPGA pathway assumes only what a CS degree already teaches. Up to 9 credits can come from CU's MS-CS, so you can mix in its Security & Ethical Hacking courses. 12 months is possible; 24 is typical.
- **Best cybersecurity picks:**
  - Old Dominion MS Cybersecurity: about $19.9k, published 1-year full-time plan.
  - Mississippi State MS Cyber Security & Operations: about $19.5k.
  - Both schools hold NSA's rare Cyber Operations designation (CAE-CO), and both programs include reverse engineering and cyber-physical security.
  - Georgia Tech OMS Cybersecurity is the cheapest top brand (about $15k), but it takes about 20 months.
- **Deepest hardware security coursework: UF EDGE's online ECE courses.** They cost about $725 per credit, but a CS grad has to take undergrad catch-up courses first. You can take them as a non-degree student and transfer up to 15 credits.
- **Money tactic:** start before January so employer tuition assistance covers two tax years, up to $10,500 tax-free.

---

## 1. Why broaden instead of doubling down on CS

### The AI hit lands on entry-level workers, not experienced engineers
- **Entry-level:** Stanford's "Canaries in the Coal Mine" (Aug 2026 update, based on ADP payroll data) finds 22–25-year-olds in AI-exposed jobs 19% below the employment trend of less-exposed peers. Young software developers are down about 20% from their late-2022 peak. Experienced workers show no comparable gap. ([paper](https://digitaleconomy.stanford.edu/app/uploads/2026/08/Canaries_August2026.pdf))
- **Programmers:** Anthropic ranks computer programmers #1 in "observed exposure": about 75% of their tasks appear in automated Claude usage. It found no rise in unemployment among exposed workers over 25. ([Anthropic](https://www.anthropic.com/research/labor-market-impacts))
- **Counter-evidence:**
  - Yale Budget Lab finds no clear AI footprint in the labor market yet. ([Yale](https://budgetlab.yale.edu/research/tracking-impact-ai-labor-market))
  - The NY Fed attributes most of the rise in young-graduate unemployment to remote work. ([NY Fed](https://libertystreeteconomics.newyorkfed.org/2026/06/remote-work-leaves-younger-workers-sidelined/))

### Official software outlook: still growing, but cut three releases in a row
- **Projections:** BLS projects software developer jobs to grow **+10.2%** over 2025–35, with a median salary of $135,980. The previous two releases projected 17.9% and 15.8%. ([BLS](https://www.bls.gov/ooh/computer-and-information-technology/software-developers.htm), [BLS MLR](https://www.bls.gov/opub/mlr/2025/article/incorporating-ai-impacts-in-bls-employment-projections.htm))
- **Exposure rating:** BLS's new AI-exposure category rates software developers "Very high." BLS says exposure isn't a forecast of job losses. ([BLS](https://www.bls.gov/emp/publications/ai-exposure-categories.htm))

### Skills AI complements vs. skills it replaces
- **"GPTs are GPTs":** programming skill is the strongest positive predictor of AI exposure. Science and critical-thinking skills are negative predictors. ([Eloundou et al.](https://arxiv.org/abs/2303.10130))
- **Anthropic's Claude Code study:** "coding agents are making a coding background less relevant." Success tracks subject-area expertise instead. **This is the core argument for adding a second field.** ([Anthropic](https://www.anthropic.com/research/claude-code-expertise?invite=1))

### Candidate fields compared

National figures are BLS 2025–35 projections with May 2025 medians. California figures are EDD 2024–34 ([CA open data](https://data.ca.gov/dataset/long-term-occupational-employment-projections)).

| Field | Target occupation | US growth / median | CA growth / median | Verdict |
|---|---|---|---|---|
| ECE: architecture / VLSI / hardware | Computer hardware engineers | +9% / $161,740 | +7.8% / $190,364 | Top pick for a hedge; small field |
| EE: power / energy | Electrical engineers | +10% / $120,630 | +7.3% / $148,072 | Fed by AI data-center power demand; bigger jump from CS |
| Robotics / embedded | (no BLS category) | — | — | Demand evidence comes only from recruiters; cyclical |
| Cybersecurity | Information security analysts | +21% / $129,180 | +29.1% / $142,449 | Strong growth; certifications matter as much as a degree |
| CS with an AI/ML focus | Data scientists | +35% / $120,230 | +37.1% / $145,554 | A bet on AI, not a hedge against it; most exposed and increasingly crowded |
| Management (MEM / MBA) | IT managers | +16% / $175,140 | +16.5% / $221,952 | Also rated "Very high" AI exposure; tech companies are flattening management |

Sources: BLS pages for [hardware engineers](https://www.bls.gov/ooh/architecture-and-engineering/computer-hardware-engineers.htm), [electrical engineers](https://www.bls.gov/ooh/architecture-and-engineering/electrical-and-electronics-engineers.htm), [information security analysts](https://www.bls.gov/ooh/computer-and-information-technology/information-security-analysts.htm), [data scientists](https://www.bls.gov/ooh/math/data-scientists.htm), [IT managers](https://www.bls.gov/ooh/management/computer-and-information-systems-managers.htm).

**Supporting signals for hardware and power:**
- Utilities are the fastest-growing sector in BLS projections (+9.8%), driven by data-center electricity demand. ([BLS release](https://www.bls.gov/news.release/ecopro.nr0.htm))
- The semiconductor industry projects a 67k-worker gap by 2030, including 12,300 master's-level engineers. This forecast comes from an industry group. ([SIA](https://www.semiconductors.org/wp-content/uploads/2023/07/SIA_July2023_ChippingAway_website.pdf))
- **Caveat:** chip design is also being automated. Synopsys claims its AI agents deliver 2–5x productivity on RTL design and verification. ([Synopsys](https://news.synopsys.com/2026-07-26-Synopsys-Showcases-Comprehensive-Autonomous-Engineering-Workflows-from-Silicon-to-Systems,-Developed-with-NVIDIA-Technology))

### How much a master's actually pays
- **Causal estimates** compare the same people's earnings before and after the degree: about +11% for engineering master's, +18–20% for CS/math, and about +10% for an MBA. ([Altonji & Zhong, NBER](https://www.nber.org/system/files/working_papers/w26959/w26959.pdf))
- **Implication:** at a California software salary, quitting to study full-time rarely pays off. The case is strongest for cheap, part-time, employer-funded programs, and for degrees that open a door you can't open otherwise (hardware, robotics, security).

---

## 2. ECE / embedded / hardware programs

| Program | Total | Fastest finish | CS-grad entry | Focus areas |
|---|---|---|---|---|
| **CU Boulder MS-ECE (Coursera)** | **$20,010** | ~12 mo (typical 24) | **Open, no application** | Embedded Linux, real-time, FPGA, IoT, power electronics, semiconductors, optics |
| Kennesaw State MSECE | ~$14.5k (my estimate) | 12–18 mo | Undergrad catch-up courses set case by case | General ECE; modest brand |
| Colorado State ME in EE / CpE | $22,410 | not stated | CS explicitly accepted (online page) | Computer architecture, hardware/software embedded, FPGA |
| Purdue online MSECE | $28,500 | usually 1 course/semester | CS preferred; math-heavy expectations | **VLSI, microelectronics**, computer engineering, power |
| Georgia Tech online MSECE | ~$34–36k | not stated | Eligible; undergrad ECE prep "encouraged" | Computer engineering, signal processing, energy |
| Missouri S&T MS CpE | $36,000 | not stated | CS usually meets requirements | Digital/VLSI, embedded |
| UF EDGE MS ECE | ~$21.8k | ~24 mo | Hard: undergrad catch-up courses required | **VLSI, analog IC, hardware security**, FPGA |
| Johns Hopkins EP ECE | $56,200 | — | Provisional until bridge courses done | Too expensive |

Georgia Tech OMSCS (about $9.5k) has Computational Perception & Robotics and Computing Systems tracks (embedded, GPU, architecture). It's the cheapest adjacent option, but the diploma says Computer Science. ([OMSCS](https://omscs.gatech.edu/specialization-computing-systems))

### CU Boulder MS-ECE: details
- **Cost:** $667 per credit × 30 credits, paid per course. Same for all residencies, no fees, no FAFSA eligibility. ([Coursera](https://www.coursera.org/degrees/msee-boulder), [CU Bursar](https://www.colorado.edu/bursar/costs/graduate-costs/coursera/professional-masters-coursera))
- **Admission:** complete one pathway specialization for credit with a 3.0 GPA. No transcripts, GRE, or even a prior degree. ([catalog](https://catalog.colorado.edu/graduate/colleges-schools/engineering-applied-science/programs-study/electrical-computer-engineering/electrical-computer-engineering-master-science-online-msece/))
- **Pace:** 6 eight-week sessions per year, capped at 15 credits per two-session semester. A 12-month finish means about 5 one-credit courses every session. ([CU EMP FAQ](https://www.colorado.edu/emp/coursera/faqs))
- **Credit sharing:** up to 9 credits from CU's MS-CS or MS-AI on Coursera count toward the degree. ([FAQ](https://www.colorado.edu/ecee/current-students/ms-ece-coursera-faq))
- **Downsides:**
  - The short 1-credit courses are less rigorous than OMSCS or UT Austin.
  - No transfer credit from other schools.
  - Some courses require hardware kits.
  - The FPGA tool (Quartus) doesn't run on macOS.

### Which pathway fits a CS grad

| Pathway | Assumes | Fit |
|---|---|---|
| **FPGA Design for Embedded Systems** (ECEA 5360–5363) | C/assembly, digital logic, basic computer architecture | **Best.** No circuits or physics needed. Needs a Terasic DE10-Lite board. ([ECEA 5360](https://www.colorado.edu/ecee/academics/online-programs/ms-ee-coursera/curriculum/embedded-systems/ecea-5360-introduction-fpga)) |
| Optical Engineering (ECEA 5600–5602) | Undergrad physics, calculus, linear algebra, MATLAB | OK if your math is solid |
| Embedding Sensors & Motors (ECEA 5340–5343) | Basic EE fundamentals, op-amps | Doable with some circuits self-study |
| Power Electronics (ECEA 5700–5703) | Circuit analysis, electronics, Bode plots | Not realistic without circuits |
| Semiconductor Devices (ECEA 5630–5632) | Physics E&M, modern physics, differential equations | Not realistic without physics |

After admission, the rest of the degree can be mostly embedded-software courses: Real-Time Embedded Systems, Advanced Embedded Linux, Embedded Interface Design, and Network Systems.

---

## 3. The ECE prerequisite gap

### What a CS degree usually covers, and what it's missing
- **Usually covered:** discrete math, computer organization, often digital logic, Calc I–II, often linear algebra.
- **Usually missing:** circuits, signals & systems, electronics, electromagnetics (E&M), differential equations, and sometimes Calc III. ABET's CS accreditation criteria don't require any of these. ([ABET CAC 2026–27](https://www.abet.org/wp-content/uploads/2025/12/2026-2027_CAC_Criteria.pdf))

### What each program family needs

| Group | Programs | Catch-up needed | Extra cost / time |
|---|---|---|---|
| A. No enforced prerequisites | CU Boulder (FPGA pathway) | None; optionally review digital logic | ~$0; admitted in 2–4 months |
| B. Computer-engineering tracks | Missouri S&T CpE, CSU CpE, Purdue CE focus | Missouri S&T: probably none ([handbook](https://ece.mst.edu/media/academic/ece/documents/handbooks/Graduate%20Handbook%20rev%2026SP%204-19-2026.pdf)). CSU/Purdue: differential equations, Calc III, linear algebra ([Purdue self-assessment](https://engineering.purdue.edu/ECE/Academics/Online/self-assessment)) | ~$150–200; 1 semester |
| C. EE-flavored | Kennesaw, UND, Georgia Tech MSECE, CSU EE | Differential equations, physics E&M, circuits with lab; maybe signals & systems | ~$600–2,300; 6–10 months |
| D. Full EE foundations | Johns Hopkins, UF | Calc III, differential equations, 2 semesters of physics, then circuits and signals bridge courses ([JHU](https://e-catalogue.jhu.edu/engineering/engineering-professionals/electrical-computer-engineering/electrical-computer-engineering-master-science/), [UF articulation](https://ece.ufl.edu/admissions/graduate/articulation/)) | ~$3.6k; 9–12 months |

### Cheapest credit-bearing ways to fill the gap
- **California community colleges** via the CVC Exchange, at $46 per unit. ([CCCCO](https://www.cccco.edu/-/media/CCCCO-Website/docs/handbook/2026-student-fee-handbook.pdf))
  - **Circuits (ENGR 260):** Cañada offers it online, but its lab is in person. Orange Coast's ENGR A285 is approved for online lab delivery. ([Cañada](https://www.canadacollege.edu/programreview/comprehensive-review-24-25/engineering-pr.pdf), [OCC](https://catalog.cccd.edu/courses/engr-a285/))
  - **Physics E&M with lab:** Coastline PHYS C280. ([CCCD](https://catalog.cccd.edu/courses/phys-c280/))
  - **Differential equations:** offered online at several colleges.
  - **Limit:** community colleges don't offer signals & systems.
- **Signals & systems and circuits bridges:**
  - Johns Hopkins EN.525.201 (Circuits, Devices & Fields) and EN.525.202 (Signals & Systems), $1,510 each, online. ([JHU tuition](https://ep.jhu.edu/admissions-aid/tuition-fees/))
  - Or ASU Online as a non-degree student: EEE 202 Circuits (~$2,320) and EEE 203 Signals (~$1,740). ([ASU](https://asuonline.asu.edu/what-it-costs))
- **Avoid Sophia, Study.com, and StraighterLine.** They don't offer these courses, and their credits aren't regionally accredited transcripts that grad schools accept. ([StraighterLine](https://www.straighterline.com/blog/degreecompletion-can-you-use-straighterline-for-graduate-school-prerequisites))

---

## 4. Cybersecurity programs

| Program | Total | Fastest finish | NSA designation | Hardware / systems fit |
|---|---|---|---|---|
| **Old Dominion MS Cybersecurity** | ~$19,860 + fees | **1 year** (published full-time plan) | CAE-CO, CAE-CD | Cyber-physical security, reverse engineering, malware, ethical hacking electives. Core includes law/policy and leadership |
| **Mississippi State MS Cyber Security & Operations** | ~$19,450 | 12 months unverified (campus norm is 2 yrs) | All three, including CAE-CO | Required courses: software reverse engineering, advanced cyber operations, cryptography, secure software engineering |
| Georgia Tech OMS Cybersecurity | ~$15k | ~17–20 mo (2 courses per term max) | Not found in NSA directory | Cyber-Physical track is power-grid and industrial-control security; hardware-security electives online |
| ASU online MCS, Cybersecurity concentration | $15,000 | ~12 mo plausible | Listed | CS degree with 3 security courses; Software Security includes binary exploitation and a CTF |
| Dakota State MS Cyber Defense | ~$14.9–18.4k | 1–1.5 yrs | CAE-CD (school also holds CAE-CO) | Operational defense and malware; little hardware |
| WGU MS Cybersecurity & Info Assurance | ~$5.1k per 6-mo term | 6–12 mo | — | Enterprise/compliance; bundles CySA+ and PenTest+ vouchers. Weakest brand |
| Purdue MS Cybersecurity & Trusted Systems | $24,750 | 12–24 mo | — | Pentesting; over budget |
| UF EDGE hardware security courses (inside the ECE MS) | ~$725 per credit | — | — | **Deepest hardware security:** Trojans, side channels, PUFs, a hands-on attack lab |

Sources: [ODU catalog](https://catalog.odu.edu/graduate/cybersecurity/cybersecurity-ms/), [ODU cost](https://online.odu.edu/cost/graduate-cost), [MSU](https://www.online.msstate.edu/mscyso), [GT OMS Cyber](https://pe.gatech.edu/degrees/cybersecurity/curriculum), [ASU](https://asuonline.asu.edu/online-degree-programs/graduate/computer-science-cybersecurity-mcs), [DSU](https://dsu.edu/programs/mscd/), [WGU](https://www.wgu.edu/online-it-degrees/cybersecurity-information-assurance-masters-program.html), [Purdue](https://www.purdue.edu/online/program/master-of-science-in-cybersecurity-and-trusted-systems/), [UF ECE online](https://ece.ufl.edu/academics/online-courses/), [NSA CAE directory](https://www.caecommunity.org/_files/ugd/fd9f4b_4248a3dadef84805adcc9d1a3bf78fd4.pdf).

**About NSA designations:** CAE-CO (Cyber Operations) requires depth in exploitation and reverse engineering. Only about 22 schools hold it. ([NSA](https://www.nsa.gov/Academics/Centers-of-Academic-Excellence/Cyber-Operations/), [Spokesman](https://www.spokesman.com/stories/2026/mar/13/ewu-earns-rare-nsa-accreditation/))

### Job market and AI exposure for security
- **Pay:**
  - Security software engineers: $274k median total compensation. ([Levels.fyi](https://www.levels.fyi/t/software-engineer/title/security-software-engineer.md))
  - California hardware and firmware security roles post high base-salary ranges. NVIDIA's hardware Root-of-Trust architect role lists $184k–$356k; Google's embedded/silicon security role lists $174k–$252k. ([NVIDIA](https://jobs.nvidia.com/careers/job/893396186854), [Google](https://jobs.anitab.org/companies/google-24698/jobs/76780640-senior-software-engineer-embedded-security-silicon))
- **Regulation is driving product-security demand.** The EU Cyber Resilience Act's reporting duty started Sept 11, 2026, and full obligations start Dec 2027. ([EC](https://digital-strategy.ec.europa.eu/en/policies/cra-summary))
- **AI exposure varies within security:**
  - Web and app pentesting is being automated. XBOW, an AI system, topped HackerOne's US leaderboard. In DARPA's AI Cyber Challenge (AIxCC), systems found 54 of 63 planted bugs. ([XBOW](https://xbow.com/blog/top-1-how-xbow-did-it), [DARPA](https://www.darpa.mil/news/2025/aixcc-results))
  - Silicon, firmware, and physical-attack work (side channels, fault injection, secure boot) needs lab equipment and hardware access, so it's less exposed. This is an inference, not a measured result.
- **Certifications matter:**
  - Employers often weight certifications and experience over degrees. ([ISC2](https://www.isc2.org/Insights/2025/06/cybersecurity-hiring-trends-study))
  - OSCP costs $1,749–2,749 and is the standard credential for offensive roles. ([OffSec](https://www.offsec.com/pricing/))
  - A master's doesn't shorten CISSP's 5-year experience requirement if your bachelor's already used the one-year waiver. ([ISC2](https://www.isc2.org/certifications/cissp/cissp-experience-requirements))
  - **Best combo:** a cheap degree plus OSCP for offensive roles. For hardware security, employers want embedded, architecture, and crypto skills plus a portfolio.

---

## 5. Speed and money tactics

- **Split payments across two tax years.** Employer education assistance (IRS Section 127) is tax-free up to $5,250 per calendar year, and unused amounts don't carry over. A program that crosses Jan 1 can use $10,500. ([26 USC 127](https://uscode.house.gov/view.xhtml?req=%28title%3A26+section%3A127+edition%3Aprelim%29))
  - The 2025 OBBBA made the student-loan repayment option permanent and indexes the cap to inflation starting in 2027.
  - California's state exclusion stays at $5,250, isn't indexed, and doesn't cover loan repayment. ([EY](https://taxnews.ey.com/news/2026-0665-california-law-largely-does-not-conform-to-obbba-provisions-affecting-compensation-and-benefits))
- **Employer payments above $5,250** can still be tax-free as a "working condition" benefit if the degree improves skills for your current job. It depends on your employer's plan. ([IRS Pub 15-B](https://www.irs.gov/publications/p15b))
- **Pre-study CU Boulder courses on Coursera Plus (~$399 per year),** then upgrade to for-credit during an enrollment window. Progress carries over. ([CU FAQ](https://www.colorado.edu/cs/academics/online-programs/mscs-coursera/faq))
- **Lifetime Learning Credit:** phases out at $80–90k MAGI for single filers, so most software engineers can't claim it. ([IRS](https://www.irs.gov/faqs/childcare-credit-other-credits/education-credits))
- **Federal loans:** Grad PLUS ended July 1, 2026, and loan limits are now prorated for less-than-full-time enrollment. Both are mostly irrelevant here, since every recommended program costs less than the $20,500 annual unsubsidized limit. ([NASFAA](https://www.nasfaa.org/uploads/documents/OB3_What_Graduate_Students_Need_to_Know.pdf))
- **California in-state public options don't help.** Chico State's online MSCS costs ~$25.5k and takes ~2 years. CSU Fullerton's MS Software Engineering costs ~$14k but takes ~22 months. ([Chico](https://rce.csuchico.edu/online-ms-computer-science/tuition-financial-aid), [CSUF](https://www.fullerton.edu/ecs/mse/programs/programcost.php))
- **California isn't part of NC-SARA,** the interstate agreement for online programs. Accredited public and nonprofit schools are exempt from California registration, but confirm each program enrolls Californians. ([BPPE](https://www.bppe.ca.gov/schools/outofstate_reg.shtml))

---

## 6. Suggested paths

1. **ECE with a security flavor (recommended):**
   - Start CU Boulder's FPGA pathway now. The next enrollment window closes around Oct 2, 2026; preview the courses on Coursera Plus first.
   - After admission, take the embedded Linux, real-time, and FPGA courses, and use the 9-credit allowance for MS-CS Security & Ethical Hacking courses.
   - Optionally add UF EDGE hardware security courses as a non-degree student.
   - Cost: ~$20k, minus up to $10.5k from employer assistance. Time: 12–24 months.
2. **Pure cybersecurity with the best offensive credibility:**
   - Old Dominion (1-year plan) or Mississippi State, both about $19.5–20k and both CAE-CO.
   - Add OSCP.
   - Georgia Tech OMS Cybersecurity if brand matters more than speed.
3. **Deeper hardware (chips/VLSI), with a longer runway:**
   - Take differential equations, Calc III, and E&M at a community college (~$200, 1 semester).
   - Then apply to Purdue ($28.5k, VLSI focus) or UF EDGE (~$21.8k plus catch-up courses).
4. **Cheapest possible credential:** WGU (~$5k per 6-month term). It's fast, but it has the weakest signal and carries some reputation risk from "degree in weeks" press coverage. ([WaPo via archive](https://archive.ph/k8Ruj))

## Open questions

- Whether Mississippi State's online degree can be finished in 12 months, and whether it charges distance fees.
- Kennesaw's actual catch-up course list for a CS grad, and whether those courses run online.
- Whether any California community college currently runs a fully online circuits lab (ENGR 260L) section. Check [search.cvc.edu](https://search.cvc.edu/).
- Whether Georgia Tech's online hardware security electives (ECE 8843, 8873) are offered every term.
- Colorado State's real practice with CS applicants to the EE (not CpE) degree. Its department handbook is stricter than its marketing page.
- Your employer's tuition policy: annual cap, reimbursement timing, and clawback if you leave.
