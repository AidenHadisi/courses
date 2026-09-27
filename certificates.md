# Valuable Certificates in Cybersecurity and AI (Sept 2026)

Research into certificates worth earning online as a working software engineer in California with a BS in Computer Science. Each credential is judged on three things: whether it builds real skill, whether it helps with hiring (job postings, résumé filters, how hiring managers see it), and whether it's a quick résumé win. It covers four kinds of credential:

- **Industry certification exams:** OSCP, AWS, and similar.
- **Online professional certificates:** Stanford Online, Maven, and similar.
- **Free training:** courses with no paid credential attached.
- **University graduate certificates** that can stack into a master's degree.

Companion doc: [masters-degree-options.md](./masters-degree-options.md) (the master's shortlist: CU Boulder MS-ECE, Old Dominion / Mississippi State cyber, and others).

Prices are 2026 list prices in USD unless noted. Cert vendors retire and rename exams often, so check before buying.

## TL;DR

- **For an experienced engineer, certs open doors but projects win offers.**
  - A randomized study of 800k+ Coursera learners found that certificates helped weak résumés and did almost nothing for strong ones. ([Wang et al.](https://arxiv.org/abs/2405.00247))
  - Oxford's analysis of 11M job postings found AI *skills* carry a 23% wage premium, versus 13% for a master's. ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0040162525000733))
  - So pick hard, hands-on, recognized exams, and pair every one with visible work.
- **Cybersecurity has the certs that most clearly matter in hiring.** CISSP and Security+ top job postings. OSCP is the standard filter for offensive-security roles.
- **In AI, only the hard cloud-vendor exams carry weight.** Most other AI certs are fluff.
- **Recommended starting set:**

| # | What | Cost | Why |
|---|---|---|---|
| 1 | **Burp Suite Certified Practitioner (BSCP)** | $99 + Burp Pro license | Cheapest real AppSec credential; plays to developer strengths |
| 2 | **AWS Certified Security – Specialty** | $300 | Top cloud-security quick win; also counts as a waiver toward CISSP's experience requirement |
| 3 | **OSCP+** (or HTB CPTS for deeper learning) | $1,749–2,749 (CPTS ~$490) | The offensive-security credential that gets past HR |
| 4 | **AWS Generative AI Developer – Professional** or **Google Professional ML Engineer** | $300 / $200 | The two AI certs that actually signal engineering skill |
| 5 | **Hack The Box Academy Silver Annual** | $490 until Oct 12 (then $550) | Best paid security training; includes exam vouchers and the AI Red Teamer path |
| Later | **CISSP** (once eligible) | $749 + $135/yr | The single most-requested security cert in job postings |

### Time-sensitive (as of Sept 26, 2026)
- **AWS ML Engineer Associate (MLA-C01):** last English exam date is **Sept 28**. The new version's beta (MLA-C02) runs from Sept 29 at **$75**. ([AWS](https://aws.amazon.com/blogs/training-and-certification/september-2026-new-offerings/))
- **Google Professional Agentic Architect:** beta costs **$120 until Sept 30** ($200 after). ([Google](https://support.google.com/cloud-certification/answer/18080541?hl=en))
- **HTB Academy Silver Annual:** **$490 until Oct 12**, then $550. ([HTB](https://www.hackthebox.com/blog/new-academy-pricing))
- **Maven AI Evals:** last 2026 cohort starts **Oct 10**. ([Maven](https://maven.com/parlance-labs/evals))

---

## 1. Do certs matter? The evidence

- **Security job postings:**
  - CyberSeek counted 514k US security listings over 12 months. The most-requested certs were CISSP (82k), Security+ (70k), CISA (52k), CISM (44k), and any GIAC cert (41k). ([CyberSeek](https://www.cyberseek.org/docs/06-02-2025_CyberSeek_June_2025.pdf))
  - A June 2026 scrape of 11.5k postings found CISSP in 19.1%, GIAC in 10.4%, Security+ in 7.7%, OSCP in 6.3%, and AWS Security in 2.7%. ([GitHub scraper](https://github.com/CarterPerez-dev/exs-cyberjob-scraper))
- **AI job postings:**
  - Informal estimates put cert mentions at only about 6–20% of postings, and none of those counts is rigorous.
  - Lightcast finds postings that list AI *skills* pay about 28% more, but it doesn't measure the effect of certs. ([Lightcast](https://lightcast.io/resources/blog/beyond-the-buzz-press-release-2025-07-23))
- **Hiring managers:**
  - In a Dice roundup (Sept 2026), hiring leaders said certs get you past automated filters but "the portfolio is the document that gets you the offer." ([Dice](https://www.dice.com/career-advice/certifications-get-you-the-interview.-portfolios-get-you-the-job))
  - Security practitioners' consensus: "OSCP is a filter, not a ceiling."
- **Where certs are concretely required:**
  - **Defense work:** DoD 8140 lists CompTIA, CISSP, CCSP, and some GIAC certs by name. ([dod8140.org](https://dod8140.org/))
  - **Consultancies:** partner programs require certified headcount. For example, Microsoft's AI Platform specialization needs staff holding AI-103. ([Microsoft Partner](https://partner.microsoft.com/en-us/partnership/specialization/ai-platform-on-microsoft-azure))
- **Salary surveys** show averages for people who hold a cert, not a raise caused by it. Skillsoft reports AWS Security Specialty holders at $203,597, CCSP at $171,524, and CISSP at $168,060. ([Skillsoft](https://www.skillsoft.com/blog/top-paying-it-certifications))

---

## 2. Cybersecurity certifications

Prep times are estimates for an experienced engineer with no prior pentesting background.

### Quick wins (weeks)
| Cert | Cost | Exam | Verdict |
|---|---|---|---|
| **Burp Suite Certified Practitioner (BSCP)** | $99/attempt + Burp Pro (~$499/yr) | 4 hr, 2 web apps, clear all 6 stages; valid 5 yrs | **Best-value AppSec cert.** Prep 1–3 months with the free Web Security Academy. ([PortSwigger](https://portswigger.net/web-security/certification/how-it-works)) |
| **AWS Security – Specialty** (SCS-C03) | $300 | 65 questions / 170 min; valid 3 yrs | **Top cloud quick win**; also on the CISSP waiver list. ([AWS](https://docs.aws.amazon.com/aws-certification/latest/security-specialty-03/security-specialty-03.html)) |
| CompTIA Security+ | $439; $150 renewal every 3 yrs | Multiple choice plus simulations | Only for federal or defense roles and HR filters. Prep 2–4 weeks. ([CompTIA fees](https://www.comptia.org/en-us/resources/ce/learn/continuing-education-renewal-fees/)) |
| TCM Practical IoT Pentest Associate (PIPA) | $249 incl. course and retake | 2 days attacking an embedded Linux device + 2 days reporting | **Cheap, credible entry point to embedded security.** ([TCM](https://certifications.tcm-sec.com/pipa/)) |

### Core offensive credential (3–6 months)
| Cert | Cost | Exam | Verdict |
|---|---|---|---|
| **OSCP+** (OffSec PEN-200) | $1,749 (90 days, 1 attempt) or $2,749/yr Learn One (2 attempts) | 24 hr hands-on + 24 hr report. The "+" expires after 3 yrs (renew via CPE credits or a higher cert) | **The HR filter for pentest and consulting roles.** ([OffSec pricing](https://www.offsec.com/pricing/), [changes](https://help.offsec.com/hc/en-us/articles/29840452210580-Changes-to-the-OSCP)) |
| **HTB CPTS** | ~$490/yr Silver Annual (includes voucher) | 10-day real-world engagement + commercial-grade report | **Teaches the most**, but weaker as an HR keyword. ([HTB](https://help.hackthebox.com/en/articles/12741732-academy-certifications)) |
| TCM PNPT | $499 incl. training | 5 days testing + 2 days report + live debrief | Good value, but optional if you're doing OSCP or CPTS. ([TCM](https://certifications.tcm-sec.com/pnpt/)) |

### Specialize (pick one track)
- **AppSec:**
  - **OSWE** (white-box source review plus exploit chaining; 48-hr exam; never expires) is the best cert for a developer moving into security. ([OffSec](https://help.offsec.com/hc/en-us/articles/360046869951-WEB-300-Advanced-Web-Attacks-and-Exploitation-OSWE-Exam-Guide))
  - After it, **HTB CWEE** (~$350 voucher) for deeper web-exploitation work.
- **Cloud and platform:** **CKA + CKS** bundle, $825. Both are hands-on command-line exams, valid 2 yrs. ([Linux Foundation](https://training.linuxfoundation.org/certification/certified-kubernetes-security-specialist/))
  - Skip AZ-500 (retired Aug 31, 2026).
  - CCSK ($445) is mainly useful because it shaves a year off CCSP's experience requirement.
- **Embedded and hardware:** there's no standard cert. Credibility comes from:
  - **PIPA**, then **OSED** (Windows exploit development).
  - Live trainings: ChipWhisperer power analysis ([link](https://www.chipwhisperer.io/live-training/cwadv/)), Praetorian's DEF CON hardware course at $2,000 ([link](https://training.defcon.org/products/bridging-the-gap-hands-on-embedded-hardware-hacking-aaron-wasserman-garrett-freibott-will-mccardell-dctlv2026)), and Joe Grand's courses ([link](https://grandideastudio.com/training/)).
  - CVEs and conference talks.
- **Red team:** Zero-Point **CRTO**, £399. Lifetime labs, unlimited attempts, and half the score is for staying undetected. ([Zero-Point](https://www.zeropointsecurity.co.uk/course/red-team-ops))
- **AI security** (all brand new, with no hiring evidence yet):
  - **OffSec OSAI** (from $1,749; 24-hr exam red-teaming LLM, agent, retrieval, and tool-calling systems) is the most credible. ([OffSec](https://www.offsec.com/courses/ai-300/))
  - Also: GIAC GAIPS ($999), CompTIA SecAI+ (launched Feb 2026).
  - ISACA AAISM requires CISSP or CISM first.

### When eligible
- **CISSP** ($749 + $135/yr):
  - Requires 5 years in 2+ of its 8 domains. **Your CS degree waives 1 year.**
  - You can pass the exam early and hold "Associate of ISC2" while you build experience.
  - Whether your software-engineering years count is decided case by case. ([ISC2](https://www.isc2.org/certifications/cissp/cissp-experience-requirements))
- **CCSP** ($599): an active CISSP waives its whole experience requirement. ([ISC2](https://www.isc2.org/certifications/ccsp/ccsp-experience-requirements))

### Skip
- **Too basic for you:** ISC2 CC, eJPT.
- **Mostly multiple choice:** PenTest+, SecurityX.
- **Retired:** AZ-500.
- **Paying for SANS/GIAC yourself** (~$8.8k course + $999 exam) is only worth it if your employer pays.

---

## 3. AI / ML certifications

The honest picture: most AI certs are entry-level badges. Only the hard, role-level cloud-vendor exams carry signal.

| Cert | Cost | Format / validity | Verdict |
|---|---|---|---|
| **AWS Generative AI Developer – Professional** (AIP-C01) | $300 | 75 questions / 180 min; 3 yrs | **Top pick if you work on AWS.** Covers retrieval-augmented generation (RAG), agents, guardrails, cost/latency tradeoffs. Rated the hardest AWS exam by people who've passed others; ~30–70 hr prep. ([AWS](https://aws.amazon.com/certification/certified-generative-ai-developer-professional/)) |
| **Google Professional ML Engineer** | $200 | 50–60 multiple choice / 2 hr; 2 yrs | **Strongest for ML fundamentals plus MLOps**; the most-recognized AI cert. ([Google](https://cloud.google.com/learn/certification/machine-learning-engineer)) |
| **AWS ML Engineer – Associate** | C01 $150 (ends Sept 28) / C02 beta $75 | 3 yrs | Good value. C02 adds Bedrock, RAG, and agents; general availability Jan 14, 2027. ([AWS](https://aws.amazon.com/certification/certified-machine-learning-engineer-associate/)) |
| **Azure AI-103** (AI Apps & Agents) | ~$165 | 1 yr, free renewal | Worth it for Azure shops and Microsoft-partner consultancies. Replaced AI-102 (retired June 30, 2026). ([Microsoft](https://learn.microsoft.com/en-us/credentials/certifications/exams/ai-103/)) |
| Google Professional Agentic Architect | $120 beta → $200 | Includes hands-on labs; general availability mid-Nov 2026 | Promising, but no track record yet. ([Google](https://support.google.com/cloud-certification/answer/18080541?hl=en)) |
| Databricks GenAI Engineer Associate / ML Professional | $200 each | 2 yrs | Only if your target employers use Databricks. ([Databricks](https://www.databricks.com/learn/certification/genai-engineer-associate)) |
| CKA / CKAD | $445 each, $825 bundle | Hands-on command line; 2 yrs | For ML-platform or MLOps infrastructure roles. |
| Anthropic Claude certs (Architect – Professional, etc.) | $125–175 | 12 months, free renewal | Carries weight at consultancies, **but only open to employees of Claude Partner Network firms.** ([Anthropic FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications)) |
| IAPP AIGP | $649–799 | 2 yrs | Only if pivoting to AI governance or policy. |

**Skip:**
- **Entry-level badges:** AWS AI Practitioner, Google Generative AI Leader, NVIDIA's associate-level exams (NCA-GENL, NCA-GENM).
- **Completion certificates, not proctored exams:** OpenAI Academy, Hugging Face.
- **Not open to you:** ISACA AAIA / AAISM (require CISA, CISM, or CISSP).
- **Retired:** AWS ML Specialty (retired Mar 2026), TensorFlow Developer (closed 2024).

**Portfolio to pair with any AI cert:**
- A RAG system with an evaluation harness.
- An agent with tracing and guardrails.
- A deployed model pipeline with monitoring.

---

## 4. Online professional certificates and free training

The credential from these programs rarely matters for someone with a CS degree. The skills and the work you produce do.

### Free and high value
| Program | What it is | Link |
|---|---|---|
| **Karpathy, Neural Networks: Zero to Hero** | Build backprop up to a GPT from scratch; YouTube videos | [karpathy.ai](https://karpathy.ai/zero-to-hero.html) |
| **Stanford CS336, Language Modeling from Scratch** | Full Spring 2026 lectures + 5 assignments. The last assignment needs rented GPUs. | [cs336.stanford.edu](https://cs336.stanford.edu/) |
| **PortSwigger Web Security Academy** | Hundreds of web-attack labs, including LLM attacks; prep for BSCP | [portswigger.net](https://portswigger.net/web-security) |
| **pwn.college (ASU)** | Belt-based exploitation dojos behind ASU's security courses; prep for OSED/OSWE | [pwn.college](https://pwn.college/) |
| fast.ai Practical Deep Learning | Solid, but 2022 content is aging | [course.fast.ai](https://course.fast.ai/) |
| Hugging Face LLM & Agents courses | Good skills; the agents cert requires passing a real benchmark | [huggingface.co/learn](https://huggingface.co/learn/agents-course/unit0/introduction) |

### Paid, worth it for structure
| Program | Cost | Format | Verdict |
|---|---|---|---|
| **Hack The Box Academy Silver Annual** | $490/yr until Oct 12, then $550 | Self-paced labs; 2 exam vouchers; AI Red Teamer path (built with Google) | **Best paid security option.** ([HTB](https://www.hackthebox.com/blog/new-academy-pricing)) |
| **Maven AI Evals for Engineers & PMs** (Husain & Shankar) | $4,200 list (promos ~25% off) | 6-week live cohort | **High value if you ship LLM features.** Great for an employer education budget. ([Maven](https://maven.com/parlance-labs/evals)) |
| **Stanford Online XCS professional certificate** (e.g. XCS224N NLP, XCS229 ML) | $2,045/course; 3 for the cert (~$6,135) | 10 weeks at 10–15 hr/wk; no academic credit | **Best-signaling affordable AI option.** ([Stanford](https://online.stanford.edu/programs/artificial-intelligence-professional-program)) |
| DeepLearning.AI ML / Deep Learning Specializations | Coursera $49/mo | Self-paced | Mostly review for a CS grad. |
| TryHackMe Premium | ~€126/yr | Self-paced | Beginner-leaning; HTB is better for you. |

### Skip
- **University-branded executive programs run by outside companies:** MIT Sloan via GetSmarter ($1.9–3.3k), Berkeley Exec Ed ($2.6–8k), eCornell, Caltech CTME, MIT Professional Education (~$18–25k).
  - Poor signal per dollar for an engineer.
  - A California State Auditor report found UC campuses gave misleading information about programs run with outside vendors. ([Auditor 2023-106](https://www.auditor.ca.gov/reports/2023-106/))
- **Google / IBM professional certificates on Coursera:** entry-level signal.
- **Learn Prompting AI red-teaming certificate** ($299–1,549): a vendor credential; HTB's AI path covers more.

---

## 5. University graduate certificates (try before a master's)

These give real transcript credit, and the best ones roll straight into a master's.

| Certificate | Cost | Stacks into | Verdict |
|---|---|---|---|
| **CU Boulder AI or Data Science certificate** (Coursera) | $6,300 (12 credits × $525) | MS-CS / MS-AI / MS-DS ($15,750 total) | **Best try-first option.** No application, pay per course, awarded automatically. ([CU](https://www.colorado.edu/cs/academics/online-programs/mscs-coursera/ai-graduate-certificate)) |
| **CU Boulder ECE certificates** (Industrial IoT, Power Electronics, Semiconductor Photonics) | $6,003 (9 credits × $667) | MS-ECE ($20,010) | IIoT fits embedded work, but it assumes circuits and lab background. ([CU ECE](https://www.colorado.edu/ecee/academics/online-programs/ms-ece-coursera/graduate-certificates)) |
| **UF Hardware & Systems Security** | ~$6,532 (9 credits × $725.75) | UF online MS ECE (pending approval) | **Best hardware-security value.** Taught by UF's hardware-security institute faculty. Whether they admit a CS grad is at ECE's discretion. ([UF](https://fics.institute.ufl.edu/hardware-system-security-certificate/)) |
| ASU AI/ML or Cybersecurity microcertificate | Program-specific (standard rate $605/credit) | ASU MCS ($15,000) | Cheap try-first path. Non-degree MCS courses (up to 12 credits) also count. ([ASU](https://asuonline.asu.edu/online-degree-programs/certificates/online-cybersecurity-certificate/)) |
| ODU Cybersecurity / Cyber Systems Security certificate | ~$8.3k (12 credits) | ODU MS-CS (documented). MS Cybersecurity not confirmed. | NSA-designated school; the ECE version includes cyber-physical security. ([ODU](https://online.odu.edu/academics/programs/cybersecurity-certificate)) |
| UT Austin CAIML | $5,000 (4 courses) | UT MSAI ($10,000) | Strong courses, but ~$1k more than applying straight to the MSAI. Same application. ([UT](https://cdso.utexas.edu/caiml)) |
| Purdue Applied AI & Cybersecurity | ~$7,425 (9 credits) | Purdue MS Cybersecurity & Trusted Systems (~$24,750) | Decent, but pricier. Skip Purdue's non-technical "Foundations of AI." |
| Harvard Extension Cyber / Data Science | $14,320 (4 courses) | ALM (~$43k) | Easy to start, but expensive per credit. |
| Stanford AI or Cybersecurity Graduate Certificate | $19.5k–27.6k | Stanford MS (only if separately admitted; ~$75k) | Real Stanford transcript credit (e.g. CS229, CS224N), but ~4× CU's price. Prestige over value. ([Stanford](https://online.stanford.edu/programs/artificial-intelligence-graduate-certificate)) |
| Johns Hopkins EP certificates | ~$22.5k | JHU EP master's (up to 5 courses) | Only if your employer pays. |

**No certificate on-ramp:**
- **Georgia Tech:** doesn't sell OMSCS courses to non-degree students, so apply to the degree directly.
- **Mississippi State:** no clear standalone cyber certificate.
- **Penn's AI certificate:** only open to Penn students and alumni.
- **UC Extension certificates** (UCSD, Berkeley, UCLA): professional certificates with no advertised master's pathway.

---

## 6. How to prepare

For each recommended cert: the official route, the courses and books people who passed actually used, practice exams or labs, free resources, and a study plan. The plans assume an experienced software engineer with no pentest background. They're estimates drawn from 2025–26 write-ups by people who passed.

**OffSec pricing** is the same for every 200/300-level course (PEN-200, WEB-300, EXP-301, AI-300): $1,749 for 90 days + 1 exam attempt, or $2,749/yr Learn One (2 attempts + Proving Grounds Practice labs). ([OffSec](https://www.offsec.com/pricing/))

**Udemy courses** list at ~$120 but are usually $15–25 on sale.

### Cybersecurity: offensive and application security

**Burp Suite Certified Practitioner (BSCP)**. Study plan: 8–12 weeks at 10–15 hr/wk.
- **Official:** The free [Web Security Academy](https://portswigger.net/web-security) *is* the course. Also PortSwigger's [how-to-prepare guide](https://portswigger.net/web-security/certification/how-to-prepare) and a free practice exam.
- **Practice:**
  - Do every Apprentice and Practitioner lab; Expert labs aren't on the exam.
  - Grind "mystery labs": random labs with the topic hidden.
  - Repeat the practice exam under time pressure.
- **Free:** [botesjuan's BSCP study notes](https://github.com/botesjuan/Burp-Suite-Certified-Practitioner-Exam-Study) (110+ labs, with scripts), Rana Khalil's YouTube walkthroughs, and PortSwigger's XSS/SQLi cheat sheets.
- **Books:** None needed. *Web Application Hacker's Handbook* (2011) is background only.
- **Tip:** Start the 30-day Burp Pro trial only in the final month.

**OSCP+**. Study plan: 16–24 weeks at 12–15 hr/wk. Roughly 8 weeks of PEN-200, 8–12 weeks of machines with a heavy Active Directory (AD) focus, then 2 weeks on OSCP-A/B/C with no hints.
- **Official:** PEN-200 plus its three practice exams (OSCP-A/B/C). The exam is 3 standalone boxes (20 pts each) plus a mandatory 40-pt AD set that starts from supplied credentials. You need 70/100. ([exam guide](https://help.offsec.com/hc/en-us/articles/360040165632-OSCP-Exam-Guide))
- **Labs:**
  - [Proving Grounds Practice](https://www.offsec.com/products/proving-grounds/) ($19/mo) is the most exam-like platform.
  - Pick boxes from [TJ Null's list](https://docs.google.com/spreadsheets/d/1dwSMIAPIam0PuRBkCiDI88pU3yzrqqHkDtBngUHNCw8/) or [LainKusanagi's list](https://www.reddit.com/r/oscp/comments/1c8pzyz/lainkusanagi_list_of_oscp_like_machines/). The LainKusanagi list separates out AD boxes.
  - People who pass typically complete 45–70 boxes.
- **Free:** IppSec's YouTube walkthroughs, Proving Grounds Play, TryHackMe free rooms.
- **Books:** *The Hacker Playbook 3*, Georgia Weidman's *Penetration Testing* (dated). Both cover fundamentals only.
- **Common path:** Do HTB CPTS or TCM's Practical Ethical Hacking first, then OSCP. ([write-up](https://www.mycyber.quest/2026/01/30/from-cpts-to-oscp-using-the-sword-to-slay-the-dragon/))

**HTB CPTS**. Study plan: 16–24 weeks at 10–15 hr/wk.
- **Official:** You must finish all 28 modules of the Penetration Tester path before the exam. Doing the Information Security Foundations and Basic Toolset paths first helps. ([HTB](https://help.hackthebox.com/en/articles/12741732-academy-certifications))
- **Practice:**
  - Do the final "Attacking Enterprise Networks" module without hints, as a mock exam.
  - Optional: Dante or Zephyr Pro Labs.
  - Learn two-hop network pivoting (e.g. with Ligolo-ng).
- **Free:** [Bruno Rocha Moura's CPTS tips](https://www.brunorochamoura.com/posts/cpts-tips/) and his report-writing posts, SysReptor report templates, IppSec's CPTS playlist.
- **Tip:** Keep a searchable methodology notebook from day one. You'll lean on it during the 10-day exam.

**TCM PNPT**. Study plan: 8–12 weeks at ~10 hr/wk, or 2–3 weeks if done after CPTS/OSCP.
- **Official:** The $499 bundle includes 5 courses: Practical Ethical Hacking (~25 hr), OSINT, External Pentest Playbook, and Linux/Windows privilege escalation. ([TCM](https://certifications.tcm-sec.com/pnpt/))
- **Practice:** Build the home AD lab the course walks through, plus the free GOAD AD lab.

**OSWE (WEB-300)**. Study plan: 12–16 weeks at 12–15 hr/wk. Your coding background is a big advantage here.
- **Exam rules:** 47h45m, 85/100 to pass. You submit one script per machine that chains all the bugs without user input. AI chatbots and automated source-code scanners are banned. ([exam guide](https://help.offsec.com/hc/en-us/articles/360046869951-WEB-300-Advanced-Web-Attacks-and-Exploitation-OSWE-Exam-Guide))
- **Prep order:** Get to BSCP-level web skills first.
- **Practice:**
  - The WEB-300 challenge labs whose names end in "Application" mirror the exam.
  - bmdyy's vulnerable Docker apps.
  - Web CTF challenges that ship with source code.
- **Free:** [snoopysecurity/OSWE-Prep](https://github.com/snoopysecurity/OSWE-Prep), [wetw0rk/AWAE-PREP](https://github.com/wetw0rk/AWAE-PREP), LiveOverflow's videos on tracing user input to dangerous functions in source code.
- **Tips:** Script every exploit in Python `requests`. Practice remote debugging in Java, .NET, Node, PHP, and Python.

### Cybersecurity: embedded and hardware

**TCM PIPA**. Study plan: 4–6 weeks at 8–10 hr/wk.
- **Official:** The $249 bundle includes the 13-hr "Beginner's Guide to IoT and Hardware Hacking" course. The exam is firmware review in a cloud VM, so no hardware is needed. ([TCM](https://certifications.tcm-sec.com/pipa/), [prep post by the exam's creator](https://tcm-sec.com/pass-pipa-certification-exam/))
- **Books:**
  - *The Hardware Hacking Handbook* (O'Flynn & van Woudenberg, No Starch, 2021).
  - *Practical IoT Hacking* (Chantzis et al., No Starch, 2021).
- **Hands-on (optional, cheap):**
  - Hardware kit: an old TP-Link router, a 3.3V USB-serial adapter, a multimeter, a cheap logic analyzer, a CH341A flash programmer.
  - Tools to learn: binwalk, Ghidra (on MIPS binaries), PulseView.
- **Free:** OWASP IoT Top 10.

**OSED (EXP-301)**. Study plan: 4–6 weeks of groundwork (x86 assembly, WinDbg, Corelan), then 12–16 weeks of EXP-301, at 12–15 hr/wk.
- **Official:** EXP-301 covers 32-bit Windows exploits: SEH overflows, egghunters, DEP/ASLR bypass, ROP. On the exam you must use WinDbg and IDA Free; Ghidra isn't allowed. ([FAQ](https://help.offsec.com/hc/en-us/articles/360053660531-OSED-Exam-FAQ))
- **Practice:** Exploit [Vulnserver](https://github.com/stephenbradshaw/vulnserver) on a Windows 10 VM, and do every "extra mile" exercise in the course.
- **Free:**
  - Exploit-writing tutorials: Corelan, FuzzySecurity.
  - OpenSecurityTraining2's x86 courses.
  - Prep repos: [nop-tech/OSED](https://github.com/nop-tech/OSED), [r0r0x-xx/OSED-Pre](https://github.com/r0r0x-xx/OSED-Pre).
  - pwn.college for fundamentals (it's Linux-focused).

### Cybersecurity: AI security

**OffSec OSAI (AI-300)**. Study plan: start only after reaching OSCP-level skills, then 10–14 weeks at 10–15 hr/wk.
- **Official:** AI-300 covers attacks on LLMs, AI agents, retrieval-augmented generation (RAG), and MCP tool integrations. OSCP-level skills are the stated prerequisite. The exam is 24 hours and open-book, and **AI assistants are allowed**. ([FAQ](https://help.offsec.com/hc/en-us/articles/46593095198740-OSAI-Advanced-AI-Red-Teaming-AI-300-FAQ))
- **Candidate tips:** The exam also includes classic pentest chains. Build and test your AI-assistant workflow in advance, and budget for model API costs. ([review](https://somecanadian.medium.com/ai-300-osai-review-my-experience-with-offsecs-ai-red-teaming-certification-13c4bd719c0a))
- **Courses:** HTB's [AI Red Teamer path](https://roadmap.hackthebox.com/changelog/full-ai-red-teamer-job-role-path-now-available), included in Silver Annual.
- **Free labs:** [PortSwigger Web LLM attacks](https://portswigger.net/web-security/llm-attacks), Lakera Gandalf, Dreadnode Crucible, Damn Vulnerable MCP Server, Microsoft AI Red Teaming Playground Labs. ([collected list](https://github.com/sonuoffsec/AI-Security-Hub))
- **Reading:** OWASP Top 10 for LLM Applications (2025) and OWASP Agentic AI Top 10.

### Cybersecurity: cloud and foundations

**AWS Security – Specialty (SCS-C03)**. Study plan: 6–8 weeks at 8–10 hr/wk, or ~4 weeks if you use AWS daily.
- **Version note:** The exam switched to C03 on Dec 2, 2025, and added GenAI/ML security. Make sure material says C03.
- **Official:** [Exam guide](https://docs.aws.amazon.com/aws-certification/latest/security-specialty-03/security-specialty-03.html) (its appendix compares C02 vs C03). The Skill Builder prep plan has a free practice question set and a paid Official Pretest.
- **Courses:**
  - [Stephane Maarek (Udemy)](https://www.udemy.com/course/ultimate-aws-certified-security-specialty/): updated for C03, ~17 hr. It assumes Solutions Architect Associate-level knowledge.
  - [Tutorials Dojo C03 video course](https://portal.tutorialsdojo.com/courses/aws-certified-security-specialty-scs-c03-video-course/).
  - [Adrian Cantrill](https://learn.cantrill.io/p/aws-certified-security-specialty) ($80): the deepest course, but still mostly C02. His C03 version is "in production."
- **Practice exams:** [Tutorials Dojo SCS-C03](https://portal.tutorialsdojo.com/product/aws-certified-security-specialty-practice-exams/). Aim for 80%+ before booking.
- **Order:** Course, then labs in a sandbox account, then Tutorials Dojo exams, then the Official Pretest. Top up from AWS docs on Bedrock security, IAM Identity Center, and Security Lake.

**CompTIA Security+ (SY0-701)**. Study plan: 3–5 weeks at 8–10 hr/wk.
- **Timing:** SY0-801 is targeted for ~Nov 17, 2026, and 701 likely retires ~May 2027. Vouchers are version-specific. Take 701 before spring 2027 rather than waiting. ([CompTIA](https://www.comptia.org/en-us/certifications/security/v8/))
- **Courses:** [Professor Messer's free video course](https://www.professormesser.com/sy0-701-certification-course/) (~15 hr; watch at 1.5×). Jason Dion (Udemy) as an alternative.
- **Books:** Chapple & Seidl, *CompTIA Security+ Study Guide SY0-701* (Sybex, 9th ed.). Or Gibson, *Get Certified Get Ahead*.
- **Practice exams:** Messer's ($30) or Dion's (6 exams). Aim for 85%+, and drill the performance-based questions.
- **Discount:** Messer sells a discounted voucher at $395. ([link](https://www.professormesser.com/discounted-comptia-security-plus-voucher/))

**CKA → CKS (Kubernetes)**. Study plan: CKA 5–7 weeks at 8–10 hr/wk; CKS 4–6 weeks right after.
- **Official:** Both exams include 2 [Killer.sh](https://killer.sh/pricing) simulator sessions, which are harder than the real exam. Single registrations don't include them. The CKA added Helm, Kustomize, and Gateway API in Feb 2025. ([LF](https://training.linuxfoundation.org/certified-kubernetes-administrator-cka-program-changes/))
- **Courses:**
  - [KodeKloud CKA](https://kodekloud.com/courses/cka-certification-course-certified-kubernetes-administrator/) (Mumshad Mannambeth) and KodeKloud CKS.
  - Kim Wüstkamp's free CKS course on YouTube. It lacks the 2024 topics (Cilium, Pod Security Standards, SBOM), so use Killercoda for those.
- **Free:** [Killercoda CKA scenarios](https://killercoda.com/course/cka), Killercoda CKS scenarios, chadmcrowell/CKA-Exercises, kubernetes.io docs (allowed in the exam).
- **Tips:** Drill imperative `kubectl` commands for speed. Do Killer.sh ~2 weeks and ~1 week before the exam.

**CISSP (when eligible)**. Study plan: 10–12 weeks at 10–12 hr/wk. Engineers should budget extra time for the risk, asset, and security-operations domains, and for the "manager mindset."
- **Books:**
  - *ISC2 CISSP Official Study Guide* (10th ed., 2024).
  - *Official Practice Tests* (4th ed.).
  - Or the more concise *Destination CISSP* (2nd ed.).
  - For the last week: Pete Zerger's *CISSP: The Last Mile* and Luke Ahmed's *How to Think Like a Manager*.
- **Courses:** Thor Pedersen ([ThorTeaches](https://thorteaches.com/cissp/) or Udemy), or Destination Certification's MasterClass.
- **Free:** Destination Certification's MindMap videos, Pete Zerger's Exam Cram + 2024 addendum on YouTube.
- **Practice exams:** [Boson ExSim](https://www.boson.com/product/exsim-max-for-cissp/) ($99), [LearnZapp](https://www.learnzapp.com/apps/isc2/cissp/). Aim for 75–80%.

**CCSK v5 / CCSP**.
- **CCSK:** 2–3 weeks. The free [CCSK v5 Prep Kit](https://cloudsecurityalliance.org/artifacts/ccsk-v5-prep-kit/) (~125-page study guide) is the exam source. The exam is open-book but timed tightly.
- **CCSP:** 5–6 weeks right after CISSP.
  - *Official Study Guide* 3rd ed. is from 2022; the 4th ed. (covering the new AI material) lands Feb 2027.
  - Pete Zerger's free CCSP Exam Cram on YouTube.

### AI / ML

**AWS Generative AI Developer – Professional (AIP-C01)**. Study plan: 8–10 weeks at 8–10 hr/wk. Weeks 1–5 are course plus labs, then work through the exam guide task by task, then the official practice exam.
- **Official:** [Exam guide](https://docs.aws.amazon.com/aws-certification/latest/ai-professional-01/ai-professional-01.html) (foundation-model integration is the biggest domain at 31%). The [Skill Builder prep plan](https://skillbuilder.aws/learning-plan/9VXVGYT38G/exam-prep-plan-aws-certified-generative-ai-developer--professional-aipc01--english/4SCMN2659K) has a free 20-question set. The official practice exam and labs need Skill Builder at $29/mo.
- **Courses:** [Maarek & Kane (Udemy)](https://www.udemy.com/course/ultimate-aws-certified-generative-ai-developer-professional/), which includes 2 mock exams. People who passed say the real exam is much harder than the course.
- **Books:** [Tutorials Dojo study-guide eBook](https://portal.tutorialsdojo.com/product/study-guide-ebook-aws-certified-generative-ai-developer-professional-aip-c01/). There's no Sybex book yet.
- **Practice exams:** The Skill Builder official exam is the most trusted. Tutorials Dojo's set got "too easy" reviews in 2026.
- **Hands-on:** Build with Bedrock Knowledge Bases, Guardrails, Agents/AgentCore, Flows, and evaluation jobs on the free tier.

**AWS ML Engineer – Associate (MLA-C02)**. Study plan: 6–8 weeks at 8 hr/wk.
- **Timing:** The C02 beta is $75 from Sept 29; general availability is Jan 14, 2027. C02 adds Bedrock, RAG, and agents. ([C02 guide](https://docs.aws.amazon.com/aws-certification/latest/machine-learning-engineer-associate-02/machine-learning-engineer-associate-02.html))
- **Courses:**
  - [Maarek & Kane "Hands On!" (Udemy)](https://www.udemy.com/course/aws-certified-machine-learning-engineer-associate-mla-c01/): updated for C02 on Sept 4, 2026, with C02 mock exams.
  - [Tutorials Dojo C02 video course](https://portal.tutorialsdojo.com/courses/aws-certified-machine-learning-engineer-associate-mla-c02-video-course/) and [practice exams](https://portal.tutorialsdojo.com/courses/aws-certified-machine-learning-engineer-associate-mla-c02-practice-exams/).
- **Books:** Chip Huyen's *Designing Machine Learning Systems* for MLOps concepts.
- **Hands-on:** SageMaker Pipelines, Model Monitor, endpoints.

**Google Professional ML Engineer (June 2026 exam)**. Study plan: 6–8 weeks at 8–10 hr/wk. One engineer who knew ML but not Google Cloud passed in ~30 days using the Skills path plus quizzes.
- **Version note:** The June 2026 refresh has less TensorFlow and more managed services and GenAI. Vertex AI is renamed Gemini Enterprise Agent Platform.
- **Official:** [Exam guide](https://services.google.com/fh/files/misc/professional_machine_learning_engineer_exam_guide_english_new.pdf). The [Skills Boost learning path](https://www.skills.google/paths/17) (17 activities; Skills Pro is $29/mo, or use the free credits).
- **Books:** The [Sybex study guide](https://www.wiley.com/en-au/Official+Google+Cloud+Certified+Professional+Machine+Learning+Engineer+Study+Guide-p-9781119944461) is from 2023, before the refresh, so use it for fundamentals only. Pair it with *Designing Machine Learning Systems*.
- **Practice exams:** Google's official sample questions are the best calibration. One candidate scored 80% on Udemy mocks but 7/17 on Google's samples. Avoid third-party banks that don't say "June 2026."

**Google Professional Agentic Architect**. Study plan: ~6 weeks at 8–10 hr/wk, build-heavy. Ship an agent built with Google's Agent Development Kit (ADK) to Agent Runtime or Cloud Run, with an evaluation set and tracing.
- **Exam format:** ~80 questions plus hands-on labs, and you must pass both. It's valid for **1 year**. ([FAQ](https://support.google.com/cloud-certification/answer/18080541?hl=en), [exam guide](https://services.google.com/fh/files/misc/professional_agentic_architect_exam_guide_english.pdf))
- **Official:** The [Skills Boost path](https://www.skills.google/paths/4525). Google says it doesn't cover the whole exam guide.
- **Free:** [Kaggle 5-Day AI Agents Intensive](https://www.kaggle.com/learn-guide/5-day-agents) (ADK, multi-agent systems, evaluation, deployment).
- **Reading:** Chip Huyen's *AI Engineering*, Anthropic's [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents).

**Azure AI-103**. Study plan: 4–6 weeks at 6–8 hr/wk. The exam can include interactive tasks, so practice in the Foundry SDK.
- **Official:** [Study guide](https://learn.microsoft.com/en-us/credentials/certifications/resources/study-guides/AI-103), plus the free [AI-103T00 Microsoft Learn course](https://learn.microsoft.com/en-us/training/courses/ai-103t00) with Foundry labs. Microsoft's free practice assessment isn't out yet.
- **Free:** John Savill's ~2-hr study cram on YouTube, and [community study guides](https://github.com/kkaminsk/AI-103-Study-Guide).

**Databricks Generative AI Engineer Associate**. Study plan: 4–6 weeks at 6–8 hr/wk.
- **Official:** [Exam guide (Mar 2026)](https://www.databricks.com/sites/default/files/2026-03/Databricks-Certified-Generative-AI-Engineer-Associate-Exam-Guide-Mar26.pdf). The Academy's [Generative AI Engineering with Databricks](https://www.databricks.com/training/catalog/generative-ai-engineering-with-databricks-1980) path; its first course is free.
- **Hands-on:** [Databricks Free Edition](https://docs.databricks.com/aws/en/getting-started/free-edition). It lacks the Agent Bricks features, so use a paid trial for those.
- **Warning:** Avoid the "exam dump" sites promoted in Reddit threads.

### Portfolio projects (pair with any AI cert)

Build plan: 4–6 weeks at 6–8 hr/wk.
- **Project 1:** A RAG system with a hand-labeled golden dataset, plus retrieval-quality and faithfulness scores that run in CI.
- **Project 2:** A tool-using agent traced in Langfuse or LangSmith, evaluated on its full sequence of steps as well as its final answer.

Resources:
- **Evals:** Hamel Husain & Shreya Shankar's free evals FAQ and LLM-as-judge guide ([index](https://hamelhusain.substack.com/p/ai-evals-for-engineers-and-product)), or their [Maven course](https://maven.com/parlance-labs/evals).
- **Agents:** Anthropic's [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) and [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents).
- **Tracing:** Langfuse's [RAG observability and evals guide](https://langfuse.com/blog/2025-10-28-rag-observability-and-evals) and [agent evaluation cookbook](https://langfuse.com/guides/cookbook/example_pydantic_ai_mcp_agent_evaluation).
- **Books:** Chip Huyen, *AI Engineering* (O'Reilly, 2025). Alammar & Grootendorst, *Hands-On Large Language Models* (O'Reilly, 2024).

---

## 7. Suggested 12-month plan

Fits the master's direction of ECE plus security:

1. **Now to Dec 2026: quick wins.**
   - BSCP (Web Security Academy prep).
   - AWS Security Specialty.
   - Lock in HTB Silver Annual at $490 before Oct 12.
2. **Jan to Jun 2027: core offensive skills.**
   - HTB CPTS path for depth, or OSCP+ if target job postings demand it.
   - In parallel, start CU Boulder's FPGA pathway toward the MS-ECE.
3. **Jul to Dec 2027: specialize.**
   - Embedded track: PIPA, plus a live hardware training.
   - Or UF's Hardware & Systems Security certificate.
   - Or OSWE if you lean toward application security.
4. **One AI cert when you need the résumé line.** AWS GenAI Developer Pro or Google PMLE, paired with a shipped RAG/agent project with evals.
5. **When eligible:** CISSP.

**Rough cost for steps 1–3 without a grad certificate:** ~$3–4k. Much of that can go on employer education assistance ($5,250 per year tax-free).

## Open questions

- **OSCP and DoD 8140:** whether OSCP is officially listed in the v2.1 matrix (sources conflict).
- **Salary data:** there's no reliable data specific to OSCP, CPTS, OSWE, BSCP, or CKS, and no rigorous count of AI-cert mentions in job postings.
- **UF certificate:** whether UF admits a CS grad to the Hardware & Systems Security certificate.
- **ODU certificate:** whether it stacks into the MS Cybersecurity.
- **Unconfirmed prices:** MIT xPRO's ML & AI certificate (may be discontinued) and exact Harvard/Emeritus exec-ed prices.
- **Prep times:** mostly estimates from practitioner write-ups.
- **Prep-resource gaps:**
  - Exact prices for Burp Pro ($449 vs $499), Tutorials Dojo, and the Destination Certification MasterClass.
  - Whether Skill Builder has an official MLA-C02 practice exam yet.
  - Whether any third-party Google PMLE question bank matches the June 2026 exam.
  - The final SY0-801 launch date.
